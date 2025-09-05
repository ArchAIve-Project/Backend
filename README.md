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

<img height="500px" alt="Architecture Diagram White BG" src="https://github.com/user-attachments/assets/2a654b29-23a3-4509-8277-ebfe76bcea41" />

> Tech Stack

We'd like to share some deeper, more technical insights about the great deal of infrastructure that supports ArchAIve's operations in this system.

## Foundational & Auxiliary Systems

To maintain consistency and system integrity, some critical services have been designed to abstract complexity that would otherwise be ubiquitous. These fundamentally core services are integral to operations. Here, we highlight just three of them that we are most proud of.

- **DatabaseInterface**:
- **LLMInterface**:
- **ThreadManager** is an internal static interface that taps on the [`APScheduler`](https://apscheduler.readthedocs.io/en/3.x/) library to facilitate asynchronous job-based processing. `AsyncProcessor` is a wrapper on the library, and simplifies the lifecycle of a single async scheduler. ThreadManager maintains one such `AsyncProcessor` instance as the default. The default processor is used in regular operations to schedule async jobs. However, ThreadManager is powerful enough to support multiple processors, each of which can spin up multiple background threads for multiple processing. Each job is assigned as a function reference; invocation can be immediate, interval-based, datetime-based or custom. As you can tell, it is a powerful piece of infrastructure in making background processing very convenient.
- **ArchSmith** is a powerful internal tracing framework custom designed to introduce much-needed transparency into the large, complex metadata generation AI pipeline. With the novel concept of loosely-linked tracers that are passed by reference along each artefact processing run, `ArchSmith` accumulates and saves "reports" across threads and simultaneous processes in a robust manner. Additionally, the **ArchSmith Visualiser** script facilitates the intuitive browsing of all `ArchSmith` tracers and reports data (that is normally in tedious JSON) with a minimalistic Flask server. Reports can be viewed in a chronological order, offering ground-breaking transparency into how each step of the AI pipeline contributed to the processing of a given artefact. This provided great aid in debugging efforts.

## AI Infrastructure

### Pipeline Chaining

The processing workflow for any artefact is largely sequential. In order to ensure consistency and integrity, we isolated similar bits of processing in classes we called "pipelines". For example, the traditional Chinese transcription workflow is encapsulated within `CCRPipeline`, captioning within `ImageCaptioning`, and classification in `ImageClassifier` just to name a few.

These smaller pipelines are joined further into larger data type processors called `MMProcessor` and `HFProcessor`. This isolates and simplifies the processing for the bipolar artefacts - meeting minutes and event photos (often referred to as human figures in internal parlance).

An overarching final pipeline invokes either processor based on the output from the custom binary classifier which helps the system determine the type of artefact. This overarching processor, called `MetadataGenerator`, makes it incredibly simple to run processing on any artefact and get real, relevant, AI-generated metadata information in return.

### ModelStore

ArchAIve features several ML models (as you saw earlier) that have been trained on custom datasets for increased domain performance. An internal service named `ModelStore` serves as a unified interface to access model weights in-memory. Based on context, it can even download model files from Google Drive if it is not present. Loading of model weights is done by passing in callback functions defined in various AI pipelines.

`ModelStore`'s main role is to ensure that models defined in a context file truly exist. Boot up is terminated if errors occur in `ModelStore` setting up. `ModelStore.defaultModels` also contains a static list of model context definitions, particularly useful when booting up a blank slate system.

`ModelStore` also makes loaded model states accessible across the system (though this is not actually done). After calling load callbacks, any piece of code can very easily retrieve a model state, which is inside its corresponding in-memory `ModelContext`. This is how all AI pipelines in the system retrieve PyTorch model states and run inference.

### Tracing

Since there's quite a bit of processing going on, it's quite difficult to determine what's happening at each stage. The complexity of code paths also makes it challenging to debug when errors in processing occur.

Thus, all the AI pipelines we mentioned earlier tap on `ArchSmith` for tracing processing runs, providing much-needed transparency. `MetadataGenerator` obtains a tracer from `ArchSmith` which is passed by reference down the call stack of pipelines and processing. Each pipeline adds its own reports to the tracer as per its needs. Each report is rich with source, description, thread, and datetime information, making debugging much easier.

Reports are persisted on an interval basis by `ArchSmith`. Tracers are completely thread-safe, so it doesn't matter where a tracer came from or where it's going. With `ArchSmith`, you can even pause/resume tracing from anywhere in the codebase.

## API Routing

Routing is carried out through Flask Blueprints. All blueprints have been neatly organised in a `./routes` sub-folder. All routes are registered in the entrypoint (`main.py`) after the setup of services.

### Identity and Access Management

Security is critical in a system like this. Sensitive user information and the storage of internal artefact data warrants the need for strict access control.

The system, at all times, must have 1 (and only 1) superuser account. This superuser account is necessary, as it has privileges such as creating new, regular accounts. During boot, if a superuser doesn't exist, one is created.

When a user logs in, their session information is stored in the encrypted Flask `session` cookie. When a request comes in, middleware decorators (that act as bouncers) like `checkSession` validate the session information. Various security checks including privileges, expiration and auth token matching is also performed. The `checkSession` decorator also, optionally, provides the internal `User` data object, making it easier to perform user-specific actions.

### Caching

There's a significant amount of data involved in ArchAIve. This is especially so as there is an uncapped amount of potential artefacts, and the metadata information associated with each can be quite large. For general rendering of data on the frontend, data retrieval can be high volume and expensive. Thus, a caching mechanism was implemented with the `cache` decorator.

The decorator stored input-output mappings - if it has seen the set of inputs before, it'll return the in-memory cached output. The cache can be invalidated in 2 ways: a Time-To-Live setting (default is 60 seconds) and a `LiteStore` attribute listener. `LiteStore` is a lightweight JSON-based persistent store that is thread-safe. If `LiteStore` validation is applied, an endpoint's `cache` decorator invalidates when the value of a specific attribute in `LiteStore` is `True`.









