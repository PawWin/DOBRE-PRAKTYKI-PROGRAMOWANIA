import os
import time

import pytest
import requests
from dotenv import load_dotenv
load_dotenv()


@pytest.mark.skipif(
    not os.getenv("E2E_SERVICE_B_URL") or not os.getenv("E2E_SERVICE_A_URL"),
    reason="E2E requires E2E_SERVICE_B_URL and E2E_SERVICE_A_URL",
)
def test_e2e_flow():
    service_b_url = os.getenv("E2E_SERVICE_B_URL")
    service_a_url = os.getenv("E2E_SERVICE_A_URL")
    image_url = os.getenv(
        "E2E_IMAGE_URL",
        "https://kaizenrent.pl/sites/default/files/styles/galeria_powiekszony/public/obraz_pgf/styl-zycia-na-przejsciu-dla-pieszych.jpg.webp?itok=4wvTFkzg",
    )

    response = requests.post(
        f"{service_b_url}/analyze_img",
        json={"url": image_url},
        timeout=10,
    )
    response.raise_for_status()
    job_id = response.json()["id"]

    deadline = time.time() + 60
    while time.time() < deadline:
        result = requests.get(f"{service_a_url}/results/{job_id}", timeout=5)
        if result.status_code == 200:
            payload = result.json()
            assert payload["id"] == job_id
            return
        time.sleep(2)

    raise AssertionError("Timed out waiting for result")
