from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # App
    app_port: int = 8282
    app_env: str = "development"
    log_level: str = "INFO"

    # Database
    mysql_url: str = "mysql+aiomysql://pagent:pagent@localhost:3316/pagent"
    postgres_url: str = "postgresql+asyncpg://pagent:pagent@localhost:5442/pagent"
    redis_url: str = "redis://localhost:6389/0"

    # Security
    cors_allowed_origins: list[str] = ["http://localhost:3000"]

    # LLM
    gemini_api_key: str = ""
    beeknoee_api_key: str = ""
    beeknoee_base_url: str = "https://api.beeknoee.com"

    # Tools
    tavily_api_key: str = ""

    # Calendar integrations
    calendar_backend: str = "log"           # "log" | "notion" | "jira" | "google"
    notion_api_key: str = ""
    notion_database_id: str = ""
    jira_base_url: str = ""
    jira_email: str = ""
    jira_api_token: str = ""
    jira_project_key: str = "PAGENT"
    google_service_account_json: str = ""   # JSON string or file path
    google_calendar_id: str = "primary"

    # Observability
    phoenix_endpoint: str = "http://localhost:6016"
    prometheus_port: int = 9290
    otel_enabled: bool = True
    otel_service_name: str = "pagent"

    # Scheduler
    scheduler_timezone: str = "UTC"
    scheduler_reminder_interval_minutes: int = 60
    scheduler_quota_reset_daily_hour: int = 0
    scheduler_quota_reset_daily_minute: int = 0
    scheduler_quota_reset_weekly_day: str = "mon"
    notification_method: str = "log"            # "log" | "webhook"
    notification_webhook_url: str = ""

    # gRPC
    base_grpc_address: str = "localhost:9091"

    # Quota Limits
    quota_daily_token_limit: int = 100_000
    quota_weekly_token_limit: int = 500_000
    quota_monthly_token_limit: int = 2_000_000
    quota_rate_limit_per_minute: int = 60


settings = Settings()
