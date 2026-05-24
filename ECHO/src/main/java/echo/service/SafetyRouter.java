package echo.service;

public class SafetyRouter {

    public boolean isSafe(int level) {
        return level < 3;
    }

    public String safeResponse(int level) {
        if (level == 3)
            return "You are not alone. Support is available.";
        if (level == 4)
            return "Your life matters. Please consider reaching out for help.";
        return "I'm here to listen.";
    }
}
