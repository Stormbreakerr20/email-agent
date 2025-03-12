def clean_html(html):
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text()