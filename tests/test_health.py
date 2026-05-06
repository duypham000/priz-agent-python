def test_health(client):
    # Act
    response = client.get("/health")

    # Assert
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == "0.1.0"
