from setuptools import setup, find_packages

setup(
    name="ai_agent",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "google-generativeai",
        "python-dotenv",
        "Pillow",
        "streamlit",
    ],
    author="Alice07071",
    author_email="boyminseo@gmail.com",
    description="AI Agent for toy analysis",
    python_requires=">=3.8",
) 