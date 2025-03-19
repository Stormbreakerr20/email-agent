import imaplib
import email
from email.header import decode_header
from dotenv import load_dotenv
import os
import json
from bs4 import BeautifulSoup
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import sys
from langchain_community.llms import OpenAI
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from datetime import datetime
from dateutil.parser import parse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
from typing import ClassVar
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP # type: ignore
import base64

#-------------------------functions----------------------------------------------------------
def clean_html(html):
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text()

def encrypt_password(password, public_key):
    rsa_key = RSA.import_key(public_key)
    cipher = PKCS1_OAEP.new(rsa_key)
    encrypted_password = cipher.encrypt(password.encode())
    return base64.b64encode(encrypted_password).decode('utf-8')

def decrypt_password(encrypted_password, private_key):
    rsa_key = RSA.import_key(private_key)
    cipher = PKCS1_OAEP.new(rsa_key)
    decrypted_password = cipher.decrypt(base64.b64decode(encrypted_password))
    return decrypted_password.decode('utf-8')

def fetch_and_append_emails(filename: str, email_id: str, email_password: str, imap_server: str):
    print("Connecting to IMAP server...", flush=True)
    mail = imaplib.IMAP4_SSL(imap_server)
    mail.login(email_id, email_password)
    mail.select("inbox")
    print("Logged in and selected inbox.", flush=True)

    filename = filename + ".json"
    # Load existing email IDs from emails5.json
    existing_ids = set()
    if os.path.exists(filename):
        with open(filename, "r") as f:
            existing_emails = json.load(f)
        existing_ids = {email["id"] for email in existing_emails}
    else:
        existing_emails = []

    _, messages = mail.search(None, "ALL")
    messages = messages[0].split()[-200:]
    print(f"Fetched {len(messages)} latest emails.", flush=True)

    new_emails = []
    for msg_id in messages:
        msg_id_str = msg_id.decode()
        if msg_id_str in existing_ids:
            continue

        print(f"Fetching email with ID: {msg_id_str}", flush=True)
        _, msg_data = mail.fetch(msg_id, "(RFC822)")
        msg = email.message_from_bytes(msg_data[0][1])
        
        subject = msg["Subject"]
        if subject is not None:
            subject, encoding = decode_header(subject)[0]
            if isinstance(subject, bytes):
                try:
                    subject = subject.decode(encoding or "utf-8")
                except LookupError:
                    subject = subject.decode("utf-8", errors="ignore")
        else:
            subject = "(No Subject)"
        
        from_ = msg.get("From")
        from_name, from_email = email.utils.parseaddr(from_)
        date = msg.get("Date")
        
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition"))
                if "attachment" not in content_disposition:
                    part_payload = part.get_payload(decode=True)
                    if part_payload:
                        try:
                            part_body = part_payload.decode()
                        except UnicodeDecodeError:
                            try:
                                part_body = part_payload.decode('latin1')
                            except UnicodeDecodeError:
                                part_body = part_payload.decode('utf-8', errors='ignore')
                        if content_type == "text/html":
                            part_body = clean_html(part_body)
                        body += part_body
        else:
            part_payload = msg.get_payload(decode=True)
            if part_payload:
                try:
                    body = part_payload.decode()
                except UnicodeDecodeError:
                    try:
                        body = part_payload.decode('latin1')
                    except UnicodeDecodeError:
                        body = part_payload.decode('utf-8', errors='ignore')
                if msg.get_content_type() == "text/html":
                    body = clean_html(body)

        new_emails.append({
            "id": msg_id_str,
            "snippet": subject,
            "source": "gmail",
            "from": from_name,
            "from_email": from_email,
            "date": date,
            "body": body
        })
        print(f"Email from {from_name} ({from_email}) on {date} fetched.", flush=True)

    # Append new emails to JSON file
    if new_emails:
        existing_emails.extend(new_emails)
        with open(filename, "w") as f:
            json.dump(existing_emails, f, indent=4)

    mail.logout()
    print("Logged out from IMAP server.", flush=True)
    return new_emails

def get_latest_n_emails(n: int, filename: str):
    if not os.path.exists(filename):
        print(f"{filename} file does not exist.")
        return []

    with open(filename, "r") as f:
        emails = json.load(f)

    # Sort emails by date in descending order
    emails.sort(key=lambda x: parse(x["date"]), reverse=True)

    # Return the latest n emails
    return emails[:n]

def search_emails_by_keyword(keyword: str, filename: str):
    # Load emails from the specified JSON file
    if not os.path.exists(filename):
        print(f"{filename} file does not exist.")
        return []

    with open(filename, "r") as f:
        emails = json.load(f)

    # Filter emails containing the keyword in subject, snippet, or body
    filtered_emails = [
        email for email in emails
        if keyword.lower() in (email.get("snippet", "").lower() or "") or
           keyword.lower() in (email.get("subject", "").lower() or "") or
           keyword.lower() in (email.get("body", "").lower() or "")
    ]

    # Sort emails by date in descending order
    filtered_emails.sort(key=lambda x: parse(x["date"]), reverse=True)

    # Return the most recent 5 emails
    return filtered_emails[:5]

def generate_reply(email_body, instruction, llm):
    print("Generating a reply for the email based on your instruction...", flush=True)
    
    messages = [
        SystemMessage(content="You are an AI email assistant. Generate professional and detailed replies. Don't include subject line.If user confirms in text to send the reply you suggested, just reply \"send\". Also your reply should always only consist of the reply to the email."),
        HumanMessage(content=f"Generate a reply to the following email:\n\n{email_body}\n\nInstruction: {instruction}")
    ]
    reply = llm.invoke(messages)

    return reply.content.strip(), messages + [reply]

def chat_with_bot(conversation_history, user_input, llm):
    print("Chatting with the bot...", flush=True)

    conversation_history.append({"role": "user", "content": user_input})

    bot_reply = llm.invoke(conversation_history)

    conversation_history.append({"role": "assistant", "content": bot_reply.content.strip()})
    return bot_reply.content.strip(), conversation_history

def send_email(to_email, subject, body, email_account, email_password, smtp_server, smtp_port):
    print(f"Sending email to {to_email} with subject: {subject}", flush=True)
    msg = MIMEMultipart()
    msg["From"] = email_account
    msg["To"] = to_email
    msg["Subject"] = subject

    msg.attach(MIMEText(body, "plain"))

    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(email_account, email_password)
        server.sendmail(email_account, to_email, msg.as_string())
    print("Email sent.", flush=True)
# -------------------------------functions end -------------------------------------------