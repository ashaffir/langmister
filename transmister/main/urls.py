from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from . import views

app_name = "main"

urlpatterns = [
    path("", views.HomeView.as_view(), name="home"),
    path(
        "user-transcriptions/",
        views.UserTranscriptionsView.as_view(),
        name="user-transcriptions",
    ),
    path("upload/", views.upload_audio, name="upload"),
]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
