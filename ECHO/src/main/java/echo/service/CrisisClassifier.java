package echo.service;

import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.util.HashMap;
import java.util.Map;

@Service
public class CrisisClassifier {

    private static final String ML_URL = "http://localhost:9000/classify";
    private final RestTemplate restTemplate = new RestTemplate();

    public int classify(String text) {
        Map<String, String> request = new HashMap<>();
        request.put("text", text);

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);

        HttpEntity<Map<String, String>> entity = new HttpEntity<>(request, headers);

        ResponseEntity<Map> response = restTemplate.postForEntity(ML_URL, entity, Map.class);
        Map<String, Object> body = response.getBody();

        if (body == null || !body.containsKey("crisis_level")) {
            throw new RuntimeException("ML service returned invalid response");
        }

        return (int) body.get("crisis_level");
    }
}
