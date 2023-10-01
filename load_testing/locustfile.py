import os
import random

from locust import HttpUser, between, events, task

IMAGES_FOLDER = "/dataset"
filenames = os.listdir(IMAGES_FOLDER)
print(filenames)


@events.init.add_listener
def on_locust_init(environment):
    environment.filenames = os.listdir(IMAGES_FOLDER)


class QuickstartUser(HttpUser):
    host = "http://44.234.254.158"
    wait_time = between(1, 5)

    @task
    def call_root_endpoint(self):
        self.client.get("/")

    @task(3)  # 3 is the random task pick probability weight
    def call_predict(self):
        filename = self.get_random_image_filename()
        image_path = f"{IMAGES_FOLDER}/{filename}"
        print(image_path)
        # Send a request to the /predict endpoint using self.client
        # Replace '/predict' with the actual endpoint you want to test
        self.client.post("/predict", files={"file": open(image_path, "rb")})

    def get_random_image_filename(self):
        return random.choice(self.environment.filenames)
