package echo.controller;

import echo.model.EchoResponse;
import echo.service.CrisisClassifier;
import echo.service.SafetyRouter;
import echo.service.LLMClient;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

@RestController
@RequestMapping("/analyze")
public class EchoController {

    private final CrisisClassifier classifier = new CrisisClassifier();
    private final SafetyRouter router = new SafetyRouter();
    private final LLMClient llm = new LLMClient();

    @PostMapping
    public EchoResponse analyze(@RequestBody Map<String, String> request) {

        long start = System.currentTimeMillis();
        String text = request.get("text");

        int level = classifier.classify(text);
        boolean safe = router.isSafe(level);

        String reply;
        if (safe) {
            reply = llm.generate(text);   // <-- direct Groq text
        } else {
            reply = router.safeResponse(level);
        }

        long latency = System.currentTimeMillis() - start;
        return new EchoResponse(level, safe, reply, latency);
    }
}
