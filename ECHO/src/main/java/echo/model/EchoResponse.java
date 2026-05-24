package echo.model;

public class EchoResponse {
    public int crisisLevel;
    public boolean safe;
    public String response;
    public long latencyMs;

    public EchoResponse(int crisisLevel, boolean safe, String response, long latencyMs) {
        this.crisisLevel = crisisLevel;
        this.safe = safe;
        this.response = response;
        this.latencyMs = latencyMs;
    }
}
