import smtplib
import email.utils
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import mimetypes

def send_email(filename, RECIPIENT):

    # Replace sender@example.com with your "From" address.
    # This address must be verified.
    SENDER = 'mehrzad.chegini@gmail.com'
    SENDERNAME = 'Mehrzad Malmirchegini'

    # Replace recipient@example.com with a "To" address. If your account
    # is still in the sandbox, this address must be verified.
    #RECIPIENT  = 'mehrzad.chegini@gmail.com,mehrzad.malmirchegini@gmail.com,rzanbaghi@gmail.com'

    # Replace smtp_username with your Amazon SES SMTP user name.
    USERNAME_SMTP = "AKIATYBTBK226EMGGNUM"

    # Replace smtp_password with your Amazon SES SMTP password.
    PASSWORD_SMTP = "BH/Pn/V4wNiEWhhcyVDvzITUJHzaSTq5nrY4u79O/Y/p"

    # (Optional) the name of a configuration set to use for this message.
    # If you comment out this line, you also need to remove or comment out
    # the "X-SES-CONFIGURATION-SET:" header below.
    CONFIGURATION_SET = "ConfigSet"

    # If you're using Amazon SES in an AWS Region other than US West (Oregon),
    # replace email-smtp.us-west-2.amazonaws.com with the Amazon SES SMTP
    # endpoint in the appropriate region.
    HOST = "email-smtp.us-east-2.amazonaws.com"
    PORT = 587

    # The subject line of the email.
    SUBJECT = 'Realtime Trading Platform - Daily Report'


    # Create message container - the correct MIME type is multipart/alternative.
    msg = MIMEMultipart('alternative')
    msg['Subject'] = SUBJECT
    msg['From'] = email.utils.formataddr((SENDERNAME, SENDER))
    msg['To'] = RECIPIENT

    # The email body for recipients with non-HTML email clients.
    BODY_TEXT = ("RTS\r\n"
                 "Daily Report - Phase 1"
                )
    msg.attach(MIMEText(BODY_TEXT, 'plain'))


    ctype, encoding = mimetypes.guess_type(filename)
    if ctype is None or encoding is not None:
        ctype = "application/octet-stream"
    maintype, subtype = ctype.split("/", 1)

    fo=open(filename)
    file = MIMEText(fo.read(),_subtype=subtype)
    fo.close()
    file.add_header("Content-Disposition", "attachment", filename=filename)
    msg.attach(file)

    # Try to send the message.
    try:
        server = smtplib.SMTP(HOST, PORT)
        server.ehlo()
        server.starttls()
        #stmplib docs recommend calling ehlo() before & after starttls()
        server.ehlo()
        server.login(USERNAME_SMTP, PASSWORD_SMTP)
        server.sendmail(SENDER, RECIPIENT, msg.as_string())
        server.close()
    # Display an error message if something goes wrong.
    except Exception as e:
        print ("Error: ", e)
    else:
        print ("Email sent!")