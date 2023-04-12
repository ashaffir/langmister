import os
import openai
import logging
from django.core.mail import EmailMultiAlternatives

from django.core.mail import send_mail
from django.conf import settings

from transmister.settings import OPENAI_KEY

logger = logging.getLogger(__file__)


def alert_admin(alert):
    """Send a message to admin

    Args:
        alert (str): The message
    """
    subject = "Toromate Alert"
    message_txt = alert
    from_email = settings.DEFAULT_FROM_EMAIL
    to_email = [settings.ADMIN_EMAIL]
    email_message = EmailMultiAlternatives(subject, message_txt, from_email, to_email)
    try:
        email_message.send()
    except Exception as e:
        logger.error(f"ERROR: email not sent (utilities.py). Reason: {e}")
        print(f"email not sent. ERROR: {e}")

    print(f">>> Alerting Admin <<<")
    logger.info(f">>> Alerting Admin <<<")


openai.api_key = OPENAI_KEY


def transcribe(audio_file):
    audio_file = open(audio_file, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file, language="he")

    user_text = f"{transcript['text']}"

    return user_text
