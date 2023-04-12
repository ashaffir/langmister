from django.contrib import admin

from main.models import Recording, Transcription


@admin.register(Recording)
class RecordAdmin(admin.ModelAdmin):
    list_display = [
        "id",
        "created",
        "language",
    ]
    ordering = ["-created"]


@admin.register(Transcription)
class TranscriptioAdmin(admin.ModelAdmin):
    list_display = [
        "id",
        "created",
        "file",
        "language",
    ]
    ordering = ("-created",)
