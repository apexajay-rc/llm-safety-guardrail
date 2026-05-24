package echo.service;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

public class LLMClient {

    private static final String API_KEY = "YOUR API KEY"
    private static final String ENDPOINT = "https://api.groq.com/openai/v1/chat/completions";
    private static final String MODEL = "llama-3.1-8b-instant";

    private final HttpClient client = HttpClient.newHttpClient();
    private final ObjectMapper mapper = new ObjectMapper();

    public String generate(String userText) {
        try {
            String body = """
            {
              "model": "%s",
              "messages": [
                { "role": "system", "content": "You are a supportive mental health assistant. Respond briefly and empathetically." },
                { "role": "user", "content": "%s" }
              ],
              "temperature": 0.3
            }
            """.formatted(MODEL, userText.replace("\"", "\\\""));

            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(ENDPOINT))
                    .header("Content-Type", "application/json")
                    .header("Authorization", "Bearer " + API_KEY)
                    .POST(HttpRequest.BodyPublishers.ofString(body))
                    .build();

            HttpResponse<String> response =
                    client.send(request, HttpResponse.BodyHandlers.ofString());

            JsonNode root = mapper.readTree(response.body());
            return root.get("choices").get(0).get("message").get("content").asText();

        } catch (Exception e) {
            return "I'm here to listen and support you.";
        }
    }
}
