# Generated by Django 4.2 on 2023-04-11 13:08

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("main", "0001_initial"),
    ]

    operations = [
        migrations.AlterField(
            model_name="record",
            name="voice_record",
            field=models.FileField(upload_to="records/"),
        ),
    ]
