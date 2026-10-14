import unittest

import app


class AirportAssistantTests(unittest.TestCase):
    def setUp(self):
        app.app.config.update(TESTING=True)
        self.client = app.app.test_client()

    def test_home_page_and_health(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'method="post"', response.data)
        self.assertIn(b'action="/chat"', response.data)
        self.assertIn(b'action="/predict/delay"', response.data)
        self.assertIn(b'action="/predict/queue"', response.data)
        self.assertNotIn(b"<script", response.data.lower())
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["status"], "ready")

    def test_chat_form_uses_flask_route(self):
        response = self.client.post("/chat", data={"message": "Where is baggage claim?"})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Baggage Reclaim", response.data)
        self.assertIn(b"Where is baggage claim?", response.data)

    def test_chat_reply(self):
        response = self.client.post("/api/chat", json={"message": "Where is baggage claim?"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("Baggage Reclaim", response.json["reply"])

    def test_chat_rejects_empty_message(self):
        response = self.client.post("/api/chat", json={"message": " "})
        self.assertEqual(response.status_code, 400)

    def test_chat_uses_fallback_for_unrecognized_message(self):
        response = self.client.post("/api/chat", json={"message": "qzxv kjh"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["intent"], "fallback")

    def test_delay_prediction(self):
        response = self.client.post(
            "/api/predict/delay",
            json={
                "airline": "Aurora Air",
                "origin": "AAA",
                "destination": "BBB",
                "scheduled_hour": 8,
                "day_of_week": 2,
                "weather": "clear",
                "traffic_level": "medium",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertGreaterEqual(response.json["estimated_delay_minutes"], 0)

    def test_delay_rejects_unknown_weather(self):
        response = self.client.post(
            "/api/predict/delay",
            json={
                "airline": "Aurora Air",
                "origin": "AAA",
                "destination": "BBB",
                "scheduled_hour": 8,
                "day_of_week": 2,
                "weather": "unknown",
                "traffic_level": "medium",
            },
        )
        self.assertEqual(response.status_code, 400)

    def test_delay_form_renders_estimate(self):
        response = self.client.post(
            "/predict/delay",
            data={
                "airline": "Aurora Air",
                "origin": "AAA",
                "destination": "BBB",
                "scheduled_hour": "8",
                "day_of_week": "2",
                "weather": "clear",
                "traffic_level": "medium",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Estimated delay:", response.data)

    def test_queue_prediction_and_validation(self):
        response = self.client.post(
            "/api/predict/queue",
            json={
                "terminal": 1,
                "hour": 8,
                "day_of_week": 2,
                "passenger_volume": 150,
                "open_lanes": 6,
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertGreaterEqual(response.json["estimated_wait_minutes"], 0)

        invalid = self.client.post(
            "/api/predict/queue",
            json={
                "terminal": 1,
                "hour": 24,
                "day_of_week": 2,
                "passenger_volume": 150,
                "open_lanes": 6,
            },
        )
        self.assertEqual(invalid.status_code, 400)

    def test_queue_form_renders_estimate(self):
        response = self.client.post(
            "/predict/queue",
            data={
                "terminal": "1",
                "hour": "8",
                "day_of_week": "2",
                "passenger_volume": "150",
                "open_lanes": "6",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Estimated wait:", response.data)


if __name__ == "__main__":
    unittest.main()
