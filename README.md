# ThreatStride AI

ThreatStride AI is an AI-powered **STRIDE threat modeling tool** designed to help identify and analyze cybersecurity threats based on a project's technical components. It uses **CySecBERT** to retrieve relevant threats from a cybersecurity dataset and **Mistral via Ollama** to generate additional context-specific threats, including risks, mitigations, investigation questions, and red flags.

### Technologies

**Python · Streamlit · PyTorch · Hugging Face Transformers · CySecBERT · Mistral · Ollama · Pandas · JSON**

### How It Works

Users enter the components of a project, such as Flutter, Android, iOS, or React. ThreatStride AI uses semantic similarity with CySecBERT to retrieve relevant STRIDE threats, then provides the retrieved context to Mistral to generate new threats. The application also calculates local **faithfulness and relevancy metrics** and provides a hallucination warning when generated results have low grounding.

> **Note:** This project was developed as a cybersecurity threat modeling prototype. The generated results should be reviewed by a security professional before being used for real-world security decisions.
