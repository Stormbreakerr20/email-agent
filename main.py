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
from pydantic import BaseModel, Field,field_validator
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
from typing import ClassVar
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP # type: ignore
import base64
# from functions import encrypt_password,decrypt_password,fetch_and_append_emails,get_latest_n_emails,search_emails_by_keyword,generate_reply,send_email,chat_with_bot,clean_html

app = FastAPI()

# Load environment variables
load_dotenv()

IMAP_SERVER = "imap.gmail.com"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
OUTLOOK_IMAP_SERVER = "outlook.office365.com"
OUTLOOK_SMTP_SERVER = "smtp.office365.com"
OUTLOOK_SMTP_PORT = 587

openai_api_key = os.getenv("OPENAI_API_KEY")
mongodb_uri = os.getenv("MONGODB_URI")
mongodb_password = os.getenv("MONGODB_PASSWORD")
private_key = base64.b64decode(os.getenv('PRIVATE_KEY'))
public_key = base64.b64decode(os.getenv('PUBLIC_KEY'))

# Connect to MongoDB
try:
    client = MongoClient(mongodb_uri, password=mongodb_password)
    db = client['email_database']
    email_collection = db['emails']
    print("Successfully connected to MongoDB.", flush=True)
except Exception as e:
    print(f"Failed to connect to MongoDB: {e}", flush=True)

llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4-turbo")

class ChatRequest(BaseModel):
    user_prompt: str
    session_id: str
    email_id: str

class EmailRequest(BaseModel):
    email_id: str

class EmailSchema(BaseModel):
    mailID: str = Field(..., description="Must be unique")
    source: str = Field(..., pattern="^(gmail|outlook)$")
    passkey: str

    @field_validator('mailID')
    @classmethod
    def mailID_must_be_unique(cls, v):
        if email_collection.find_one({"mailID": v}):
            raise ValueError('mailID must be unique')
        return v

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

def fetch_and_append_emails(filename: str,email_id:str,email_password:str):
    print("Connecting to Gmail IMAP server...", flush=True)
    mail = imaplib.IMAP4_SSL(IMAP_SERVER)
    mail.login(email_id, email_password)
    mail.select("inbox")
    print("Logged in and selected inbox.", flush=True)

    filename = filename + ".json"
    # Load existing email IDs from emails5.json
    existing_ids = set()
    if os.path.exists(filename):
        with open( filename, "r") as f:
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
    print("Logged out from Gmail IMAP server.", flush=True)
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

# Call the function
# emails = fetch_and_append_emails()

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


# Function to generate a reply to an email using OpenAI's GPT model
def generate_reply(email_body, instruction):
    print("Generating a reply for the email based on your instruction...", flush=True)
    
    messages = [
        SystemMessage(content="You are an AI email assistant. Generate professional and detailed replies. Don't include subject line.If user confirms in text to send the reply you suggested, just reply \"send\". Also your reply should always only consist of the reply to the email."),
        HumanMessage(content=f"Generate a reply to the following email:\n\n{email_body}\n\nInstruction: {instruction}")
    ]
    reply = llm.invoke(messages)

    return reply.content.strip(), messages + [reply]


def chat_with_bot(conversation_history, user_input):
    print("Chatting with the bot...", flush=True)

    conversation_history.append({"role": "user", "content": user_input})

    llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4-turbo")  # FIXED
    bot_reply = llm.invoke(conversation_history)

    conversation_history.append({"role": "assistant", "content": bot_reply.content.strip()})
    return bot_reply.content.strip(), conversation_history

def send_email(to_email, subject, body,email_account:str,email_password:str):
    print(f"Sending email to {to_email} with subject: {subject}", flush=True)
    msg = MIMEMultipart()
    msg["From"] = email_account
    msg["To"] = to_email
    msg["Subject"] = subject

    msg.attach(MIMEText(body, "plain"))

    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(email_account, email_password)
        server.sendmail(email_account, to_email, msg.as_string())
    print("Email sent.", flush=True)
# -------------------------------functions end -------------------------------------------

@app.post("/create")
async def create_mail(request: EmailSchema):
    try:
        # Check if the passkey is correct by connecting to the email server
        if request.source == "gmail":
            server = imaplib.IMAP4_SSL(IMAP_SERVER)
        elif request.source == "outlook":
            server = imaplib.IMAP4_SSL(OUTLOOK_IMAP_SERVER)
        else:
            raise HTTPException(status_code=400, detail="Invalid email source")

        try:
            server.login(request.mailID, request.passkey)
            server.logout()
        except imaplib.IMAP4.error:
            raise HTTPException(status_code=400, detail="Invalid passkey or unable to connect to the email server")
        
        # Encrypt the passkey
        encrypted_passkey = encrypt_password(request.passkey, public_key)
        # Prepare the email data
        email_data = {
            "mailID": request.mailID,
            "source": request.source,
            "passkey": encrypted_passkey
        }
        # Insert the email data into the database
        result = email_collection.insert_one(email_data)
        print("Inserted ID:", result.inserted_id)
        return {"status": "success", "message": "Email data saved successfully"}
    except DuplicateKeyError:
        raise HTTPException(status_code=400, detail="mailID must be unique")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/gmail/startup")
async def gmail_startup(request: EmailRequest):
    try:
        filename = request.email_id.split('@')[0] + "_gmail"
        email_data = email_collection.find_one({"mailID": request.email_id})
        if not email_data:
            raise HTTPException(status_code=400, detail="Email ID not found in the database")
        temp_passkey = decrypt_password(email_data["passkey"], private_key)
        new_emails = fetch_and_append_emails(filename, request.email_id, temp_passkey)
        del temp_passkey
        return {"status": "success", "message": f"Fetched and appended {len(new_emails)} emails for {request.email_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

conversation_history = {}

@app.post("/gmail/chat")
async def gmail_chat(request: ChatRequest):
    try:
        # print(request.dict())
        user_prompt = request.user_prompt
        session_id = request.session_id
        email_id = request.email_id
        filename = email_id.split('@')[0] + "_gmail.json"
        
        if session_id not in conversation_history:
            conversation_history[session_id] = []

        intent_message = [
            SystemMessage(content='''You are an AI assistant. Determine the intent of the user's request and decide which function to call."
                You are an intent parser. Given a user query, output JSON with:
                    - "intent" (one of:show_latest_n, search, send_generated_reply,reply, chat,generate_reply)
                    - Required parameters: 
                        - "filename" (always required, derived from email account)
                        - "number" (for last N emails),
                        - "keyword" (for search),
                        - "to_email", "body" and "subject" (for send_reply),
                        - "email_body" and "instruction" (for generate_reply),
                        - Any other relevant field.

                    Respond only with JSON.
                    -Usage of function for reference:
                        - show_latest_n_emails(n, filename): Returns the latest N emails from the specified JSON file.
                        - search_emails_by_keyword(keyword, filename): Returns the emails containing the keyword in the specified JSON file.    
                        - generate_reply(email_body, instruction): Generates a reply to the email based on the instruction.
                        - send_generated_reply(to_email, subject, body): when user confirms, Sends an email to the specified email address with the subject and body.
                        - chat_with_bot(conversation_history, user_input): Chat with the bot using the conversation history and user input.
                          '''),
            HumanMessage(content=f"User prompt: {user_prompt}")
        ]
        intent_reply = llm.invoke(intent_message)
        intent = intent_reply.content.strip().lower()
        intent_data = json.loads(intent)
        conversation_history[session_id].append({"role": "assistant", "content": intent_reply.content.strip()})
        print("intent:", intent)
        if "show_latest_n" in intent:
            n = intent_data.get("number", 5)  # Default to fetching the latest 5 emails
            latest_emails = get_latest_n_emails(n, filename)
            response = json.dumps(latest_emails, indent=4)
            conversation_history[session_id].append({"role": "assistant", "content": response})
        elif "search" in intent:
            keyword = intent_data.get("keyword", "")
            searched_emails = search_emails_by_keyword(keyword, filename)
            response = json.dumps(searched_emails, indent=4)
            conversation_history[session_id].append({"role": "assistant", "content": response})
        elif "generate_reply" in intent:
            email_body = user_prompt.split("generate reply for ")[-1]
            instruction = "Please generate a professional reply."
            reply_content, _ = generate_reply(email_body, instruction)
            response = reply_content
            conversation_history[session_id].append({"role": "assistant", "content": response})
        elif "send_generated_reply" in intent:
            email_data = email_collection.find_one({"mailID": request.email_id})
            if not email_data:
                raise HTTPException(status_code=400, detail="Email ID not found in the database")
            temp_passkey = decrypt_password(email_data["passkey"], private_key)
            to_email = intent_data.get("to_email", "")
            subject = intent_data.get("subject", "")
            body = intent_data.get("body", "")
            send_email(to_email, subject, body,email_id,temp_passkey)
            response = f"Email sent to {to_email} with subject: {subject} and body: {body}"
            conversation_history[session_id].append({"role": "assistant", "content": response})
            del temp_passkey
        else:
            response, conversation_history[session_id] = chat_with_bot(conversation_history[session_id], user_prompt)
        
        # Convert response to human-readable form using GPT-4
        human_readable_message = [
            SystemMessage(content="Convert the following response to a human-readable form only if necessary,else return the message itself:"),
            HumanMessage(content=response)
        ]
        human_readable_reply = llm.invoke(human_readable_message)
        human_readable_response = human_readable_reply.content.strip()
        
        return {"status": "success", "response": human_readable_response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))