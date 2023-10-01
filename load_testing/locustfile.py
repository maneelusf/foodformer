import os
import random
from typing import Optional

from locust import HttpUser, Response, between, events, task
from locust.env import Environment

IMAGES_FOLDER = "/dataset"

# ...


@events.init.add_listener
def on_locust_init(environment: Environment) -> None:
    environment.filenames = os.listdir(IMAGES_FOLDER)


class QuickstartUser(HttpUser):
    host: str = "http://44.234.254.158"
    wait_time = between(1, 5)

    @task
    def call_root_endpoint(self) -> None:
        self.client.get("/")

    @task(3)
    def call_predict(self) -> Optional[Response]:
        filename = self.get_random_image_filename()
        image_path = f"{IMAGES_FOLDER}/{filename}"
        print(image_path)
        response = self.client.post("/predict", files={"file": open(image_path, "rb")})
        print("Response status code:", response.status_code)
        print("Response text:", response.text)
        return response

    def get_random_image_filename(self) -> str:
        return random.choice(self.environment.filenames)
