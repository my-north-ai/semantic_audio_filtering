from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    OPENAI_KEY: str
    HF_TOKEN: str
    model_config = SettingsConfigDict(env_file=".env")