<img height="250px" alt="ArchAIve Backend Logo" src="https://github.com/user-attachments/assets/f46930d3-57eb-44f3-99cf-6e0c3d840958" />

![Python Badge](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue)
![Flask Badge](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![JSON Badge](https://img.shields.io/badge/json-5E5C5C?style=for-the-badge&logo=json&logoColor=white)
![PyTorch Badge](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Firebase Badge](https://img.shields.io/badge/firebase-ffca28?style=for-the-badge&logo=firebase&logoColor=black)
![Google SMTP](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)
![OpenAI API](https://img.shields.io/badge/ChatGPT-74aa9c?style=for-the-badge&logo=openai&logoColor=white)
![AWS](https://img.shields.io/badge/Amazon_AWS-FF9900?style=for-the-badge&logo=amazonaws&logoColor=white)

# ArchAIve Backend

<img height="500px" alt="Screenshot of ArchAIve Website Homepage" src="https://github.com/user-attachments/assets/889fedc3-d0ec-4388-b6da-e6212d24d7ac" />

**ArchAIve is an AI-powered artefact digitisation platform for the preservation and proliferation of heritage and culture.**

With custom-built and highly specialised AI models and ML pipelines, the platform enables the use of AI to automate interpretation and documentation of historical artefacts through a streamlined experience.

Since it was the team's first time developing a solution for a real-world client, we poured in much effort into making a robust, powerful, and efficient system design. The result is truly remarkable - ArchAIve is a project bigger than any we've done before.

The ArchAIve team comprises of:
- [Prakhar Trivedi](https://github.com/Prakhar896) (Team Lead, Image Captioning, Automatic Categorisation, User Management)
- [Joon Jun Han](https://github.com/JunHammy) (Traditional Chinese OCR, AI Transcription Processing, Catalogue Browsing)
- [Toh Zheng Yu](https://github.com/ZyuT0h) (Image Classification, Archivus Chatbot, Data Import & Processing)

For a more general overview of ArchAIve, see [the organisation README](https://github.com/ArchAIve-Project/.github/blob/main/profile/README.md).

# Problem & Solution

Thousands of scanned Chinese historical artefacts — such as handwritten **meeting minutes in Traditional Chinese calligraphy** and **photographs from key historical eras** — remain difficult to interpret, catalogue, and search. Manual transcription and face recognition is tedious, slow, and error-prone, while no unified system exists to manage these artefacts digitally.

ArchAIve is a platform that addresses these challenges by combining advanced text recognition, face recognition, and digital cataloguing in a unified system. It streamlines complex AI pipelines into an accessible platform, enabling efficient preservation, understanding, and management of historical artefacts without compromising performance.

# Backend Introduction

ArchAIve's backend system is highly complex, intricate, and powerful. It is an API that targets the problem statement dead-on. With several layers of nuanced subsystems, processes, abstractions and workflows, the Flask-based backend API has been designed from the ground up to be efficient, idempotent and secure.

The core challenge with the digitisation of anything is the various jobs that need to be done. Scanning in meeting minutes, captioning pictures, writing out digital transcriptions are just some of the many individual tasks that make artefact digitisation incredibly tedious and slow. The goal of the backend system, which houses the core business logic, was to streamline all tasks into a highly clean, simple and efficient singular workflow that can be triggered.

Tapping on many cutting-edge libraries, this system features asynchronous job management, idempotent API service with Flask, model inference with PyTorch, pipeline-format AI workflows, custom cloud-based database management, and several auxiliary services that aid processes.

# Technical Overview

We'd like to share some deeper, more technical insights about the great deal of infrastructure that supports ArchAIve's operations in this system.

## Foundational & Auxiliary Systems

To maintain consistency and system integrity, some critical services have been designed to abstract complexity that would otherwise be ubiquitous. These fundamentally core services are integral to operations.

**Universal** contains some standard data and methods that are used across the system. This includes things like `copyright`, datetime handling, PyTorch device retrieval and file upload limits.

**Encryption** provides convenient SHA256 hashing and encoding capabilities. This results in good and consistent handling of things like user passwords and other sensitive information.

**FileOps** is a static interface the abstracts away common `os` and file operations in a safe, convenient manner. This is foundational to ensuring filesystem consistency, especially as the system features a lot of file management.

**Logger** is a thread-safe logging interface for convenient, reliable logging to persistent storage. A simple `log` method outputs a given message into a pre-set logs file. The interface has been designed to be infallible, ensuring system integrity. The interface also features an interactive mode to browse stored logs, including the ability to filter by log "tags". Each log is tagged with UTC datetime information as well.

**ThreadManager** is an internal static interface that taps on the [`APScheduler`](https://apscheduler.readthedocs.io/en/3.x/) library to facilitate asynchronous job-based processing. `AsyncProcessor` is a wrapper on the library, and simplifies the lifecycle of a single async scheduler. ThreadManager maintains one such `AsyncProcessor` instance as the default. The default processor is used in regular operations to schedule async jobs. However, ThreadManager is powerful enough to support multiple processors, each of which can spin up multiple background threads for multiple processing. Each job is assigned as a function reference; invocation can be immediate, interval-based, datetime-based or custom. As you can tell, it is a powerful piece of infrastructure in making background processing very convenient.

**ArchSmith** is a powerful internal tracing framework custom designed to introduce much-needed transparency into the large, complex metadata generation AI pipeline. With the novel concept of loosely-linked tracers that are passed by reference along each artefact processing run, ArchSmith accumulates and saves "reports" across threads and simultaneous processes in a robust manner. Additionally, the ArchSmith Visualiser script facilitates the intuitive browsing of all ArchSmith tracers and reports data (that is normally in tedious JSON) with a minimalistic Flask server. Reports can be viewed in a chronological order, offering ground-breaking transparency into how each step of the AI pipeline contributed to the processing of a given artefact. This provided great aid in debugging efforts.




