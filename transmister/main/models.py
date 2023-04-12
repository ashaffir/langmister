import uuid
from datetime import datetime
from django.db import models
from django.urls.base import reverse


def recording_path(instance, filename):
    return f"recordings/{instance.session}/{instance.pk}.wav"


class Recording(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    created = models.DateTimeField(auto_now_add=True)
    session = models.IntegerField()
    voice_recording = models.FileField(upload_to=recording_path)
    language = models.CharField(max_length=50, null=True, blank=True)

    class Meta:
        verbose_name = "Record"
        verbose_name_plural = "Records"

    def __str__(self):
        return str(self.id)

    def get_absolute_url(self):
        return reverse("main:record_detail", kwargs={"id": str(self.id)})


class Transcription(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    created = models.DateTimeField(auto_now_add=True)
    session = models.IntegerField()
    file = models.FileField()
    language = models.CharField(max_length=50, null=True, blank=True)

    class Meta:
        verbose_name = "Transcription"
        verbose_name_plural = "Transcriptions"

    def __str__(self):
        return str(self.id)

    def get_file_name(self):
        return self.file.name.split("/")[-1]

    def get_content(self):
        with open(self.file.name, "r") as f:
            content = f.readlines()
        return content
