from django.apps import AppConfig
from .utils import load_model


class HelloAzureConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'hello_azure'

    def ready(self):
        load_model()
        return super().ready()
