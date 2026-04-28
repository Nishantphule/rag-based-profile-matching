"""
One-shot dataset generator for the RAG Profile Matching assignment.

Produces:
    data/resumes/*.txt              (32 diverse resumes)
    data/job_descriptions/*.txt     (6 job descriptions)

Run:
    python scripts/generate_dataset.py
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from textwrap import dedent

random.seed(7)

ROOT = Path(__file__).resolve().parents[1]
RESUMES_DIR = ROOT / "data" / "resumes"
JD_DIR = ROOT / "data" / "job_descriptions"
RESUMES_DIR.mkdir(parents=True, exist_ok=True)
JD_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Resume templates
# ---------------------------------------------------------------------------

RESUMES: list[dict] = [
    # ---------------- Software / ML ----------------
    {
        "name": "Aarav Sharma",
        "title": "Senior Machine Learning Engineer",
        "email": "aarav.sharma@example.com",
        "phone": "+91-98765-43210",
        "location": "Bangalore, India",
        "years": 8,
        "summary": (
            "Senior ML Engineer with 8 years building large-scale recommendation "
            "and NLP systems. Led a team of 6 engineers to ship a transformer-based "
            "search ranker serving 40M users."
        ),
        "skills": [
            "Python", "PyTorch", "TensorFlow", "Transformers", "LLMs",
            "Machine Learning", "Deep Learning", "MLOps", "AWS SageMaker",
            "Kubernetes", "Docker", "SQL",
        ],
        "experience": [
            ("Staff ML Engineer", "Flipkart", "2022 - Present",
             "Designed and shipped a transformer-based product ranker that lifted "
             "GMV by 6.2%. Owned the offline + online evaluation pipeline."),
            ("Senior ML Engineer", "Swiggy", "2019 - 2022",
             "Built ETA prediction models in PyTorch reducing MAE by 18%. "
             "Productionized models with Kubernetes + Triton."),
            ("ML Engineer", "Myntra", "2017 - 2019",
             "Worked on visual search with CNN embeddings and FAISS."),
        ],
        "education": [
            ("M.Tech, Computer Science", "IIT Bombay", "2017"),
            ("B.Tech, Information Technology", "NIT Trichy", "2015"),
        ],
        "projects": [
            "Open-source contributor to Hugging Face Transformers (3 PRs merged).",
            "Built a RAG-based internal documentation assistant on LangChain + Chroma.",
        ],
    },
    {
        "name": "Priya Iyer",
        "title": "Machine Learning Engineer",
        "email": "priya.iyer@example.com",
        "phone": "+91-90000-11111",
        "location": "Hyderabad, India",
        "years": 5,
        "summary": (
            "ML Engineer with 5 years of experience focused on NLP, LLM fine-tuning "
            "and retrieval systems."
        ),
        "skills": [
            "Python", "PyTorch", "Hugging Face", "LangChain", "RAG",
            "Vector Databases", "Pinecone", "ChromaDB", "FastAPI", "AWS",
            "Machine Learning", "NLP",
        ],
        "experience": [
            ("ML Engineer", "Razorpay", "2021 - Present",
             "Fine-tuned Llama-2 7B on internal support tickets; deployed a RAG "
             "assistant cutting agent handle time by 25%."),
            ("Data Scientist", "Freshworks", "2020 - 2021",
             "Built intent classifiers (BERT) and entity extractors for chatbots."),
        ],
        "education": [
            ("B.E., Computer Science", "BITS Pilani", "2020"),
        ],
        "projects": [
            "Public blog series on building RAG with ChromaDB (15K reads).",
        ],
    },
    {
        "name": "Rohan Kapoor",
        "title": "Junior Data Scientist",
        "email": "rohan.kapoor@example.com",
        "phone": "+91-90909-90909",
        "location": "Pune, India",
        "years": 2,
        "summary": "Data scientist early in career, comfortable with classical ML and analytics.",
        "skills": [
            "Python", "scikit-learn", "Pandas", "SQL", "Tableau",
            "Statistics", "Machine Learning",
        ],
        "experience": [
            ("Data Scientist", "Mu Sigma", "2023 - Present",
             "Built churn models for a US telecom client (XGBoost, AUC 0.81)."),
            ("Data Analyst Intern", "Tata 1mg", "2022 - 2023",
             "SQL dashboards for marketing funnel analysis."),
        ],
        "education": [
            ("B.Sc., Statistics", "Fergusson College", "2023"),
        ],
        "projects": [
            "Kaggle Titanic top 12% (notebook).",
        ],
    },
    {
        "name": "Sara Mehta",
        "title": "AI Research Engineer",
        "email": "sara.mehta@example.com",
        "phone": "+91-91234-56789",
        "location": "Bangalore, India",
        "years": 6,
        "summary": (
            "Research-leaning ML engineer with publications at NeurIPS and ACL. "
            "Specializes in retrieval-augmented generation and evaluation."
        ),
        "skills": [
            "Python", "PyTorch", "JAX", "Transformers", "RAG",
            "LLMs", "Information Retrieval", "FAISS", "Weaviate",
            "Distributed Training", "Deep Learning",
        ],
        "experience": [
            ("Research Engineer", "Google DeepMind (contract)", "2022 - Present",
             "Worked on long-context retrieval for code generation."),
            ("Applied Scientist", "Amazon Alexa AI", "2019 - 2022",
             "Improved slot filling accuracy by 9% using contrastive pretraining."),
        ],
        "education": [
            ("M.S., Machine Learning", "Carnegie Mellon University", "2019"),
            ("B.Tech, CSE", "IIT Delhi", "2018"),
        ],
        "projects": [
            "First-author NeurIPS 2023 paper on dense retrieval evaluation.",
        ],
    },
    {
        "name": "Vikram Nair",
        "title": "Backend Engineer",
        "email": "vikram.nair@example.com",
        "phone": "+91-99887-66554",
        "location": "Chennai, India",
        "years": 7,
        "summary": (
            "Backend engineer with deep experience in distributed systems and "
            "high-throughput Python services."
        ),
        "skills": [
            "Python", "Go", "FastAPI", "Django", "PostgreSQL",
            "Redis", "Kafka", "Kubernetes", "AWS", "System Design",
        ],
        "experience": [
            ("Senior Backend Engineer", "Postman", "2021 - Present",
             "Owned the runtime service handling 12B requests/month."),
            ("Backend Engineer", "Zerodha", "2017 - 2021",
             "Built order management subsystems in Go."),
        ],
        "education": [
            ("B.Tech, ECE", "VIT Vellore", "2017"),
        ],
        "projects": [
            "Maintainer of an internal Python OpenTelemetry library.",
        ],
    },
    {
        "name": "Ananya Verma",
        "title": "Full Stack Developer",
        "email": "ananya.verma@example.com",
        "phone": "+91-93333-22222",
        "location": "Gurgaon, India",
        "years": 4,
        "summary": "Full-stack developer comfortable across React and Node.js.",
        "skills": [
            "JavaScript", "TypeScript", "React", "Next.js", "Node.js",
            "Express", "MongoDB", "PostgreSQL", "AWS", "Docker",
        ],
        "experience": [
            ("Full Stack Developer", "Zomato", "2022 - Present",
             "Owned the merchant dashboard React app and its Node.js BFF."),
            ("Software Engineer", "Paytm", "2020 - 2022",
             "Worked on the merchant onboarding KYC flow."),
        ],
        "education": [
            ("B.Tech, CSE", "DTU Delhi", "2020"),
        ],
        "projects": [
            "Side project: open-source Markdown-based blog engine (1.2K stars).",
        ],
    },
    {
        "name": "Karthik Reddy",
        "title": "Data Engineer",
        "email": "karthik.reddy@example.com",
        "phone": "+91-94444-55555",
        "location": "Hyderabad, India",
        "years": 6,
        "summary": "Data engineer specializing in real-time pipelines and data lakes.",
        "skills": [
            "Python", "Scala", "Spark", "Kafka", "Airflow",
            "Snowflake", "Databricks", "AWS", "SQL", "dbt",
        ],
        "experience": [
            ("Senior Data Engineer", "Walmart Labs", "2021 - Present",
             "Migrated 200+ Airflow DAGs to a Spark-on-Kubernetes platform."),
            ("Data Engineer", "PhonePe", "2018 - 2021",
             "Built the fraud detection feature store on Kafka + Flink."),
        ],
        "education": [
            ("B.Tech, CSE", "IIIT Hyderabad", "2018"),
        ],
        "projects": [
            "Talk at PyData Hyderabad on idempotent stream processing.",
        ],
    },
    {
        "name": "Neha Singh",
        "title": "Product Designer",
        "email": "neha.singh@example.com",
        "phone": "+91-95555-66666",
        "location": "Mumbai, India",
        "years": 5,
        "summary": "Product designer focused on B2B SaaS UX and design systems.",
        "skills": [
            "Figma", "Sketch", "Prototyping", "Design Systems",
            "User Research", "Accessibility", "Webflow",
        ],
        "experience": [
            ("Senior Product Designer", "Freshworks", "2021 - Present",
             "Owns the Freshchat agent console design system."),
            ("Product Designer", "Browserstack", "2019 - 2021",
             "Redesigned the live testing dashboard."),
        ],
        "education": [
            ("B.Des, Visual Communication", "NID Ahmedabad", "2019"),
        ],
        "projects": ["Speaker, UXIndia 2023."],
    },
    {
        "name": "Arjun Patel",
        "title": "DevOps Engineer",
        "email": "arjun.patel@example.com",
        "phone": "+91-96666-77777",
        "location": "Ahmedabad, India",
        "years": 6,
        "summary": "DevOps and platform engineer with cloud-native expertise.",
        "skills": [
            "Kubernetes", "Terraform", "AWS", "GCP", "Helm",
            "ArgoCD", "Prometheus", "Grafana", "Python", "Bash",
        ],
        "experience": [
            ("Senior DevOps Engineer", "Cred", "2021 - Present",
             "Owned the multi-region EKS platform; drove cost down 30%."),
            ("DevOps Engineer", "Practo", "2018 - 2021",
             "Built CI/CD with ArgoCD for 80+ services."),
        ],
        "education": [
            ("B.E., IT", "L.D. College of Engineering", "2018"),
        ],
        "projects": [
            "OSS contributor to kube-prometheus-stack helm chart.",
        ],
    },
    {
        "name": "Meera Krishnan",
        "title": "Computer Vision Engineer",
        "email": "meera.krishnan@example.com",
        "phone": "+91-97777-88888",
        "location": "Bangalore, India",
        "years": 7,
        "summary": "Computer vision engineer working on autonomous robotics.",
        "skills": [
            "Python", "C++", "PyTorch", "OpenCV", "Computer Vision",
            "ROS", "TensorRT", "CUDA", "Deep Learning", "Machine Learning",
        ],
        "experience": [
            ("CV Lead", "Ati Motors", "2020 - Present",
             "Owned perception stack for autonomous industrial vehicles."),
            ("CV Engineer", "Mahindra Electric", "2017 - 2020",
             "Built lane and obstacle detection models."),
        ],
        "education": [
            ("M.S., Robotics", "ETH Zurich", "2017"),
            ("B.E., Mechanical", "PSG Tech", "2015"),
        ],
        "projects": [
            "Built an open-source ROS2 perception toolkit.",
        ],
    },
    {
        "name": "Rahul Bose",
        "title": "Data Scientist",
        "email": "rahul.bose@example.com",
        "phone": "+91-98888-99999",
        "location": "Kolkata, India",
        "years": 9,
        "summary": (
            "Data scientist with 9 years across fintech and e-commerce. Strong on "
            "experimentation, causal inference and time-series."
        ),
        "skills": [
            "Python", "R", "SQL", "Causal Inference", "A/B Testing",
            "Machine Learning", "Time Series", "PyMC", "Tableau",
        ],
        "experience": [
            ("Principal Data Scientist", "PolicyBazaar", "2020 - Present",
             "Owns experimentation platform; runs 200+ A/B tests/year."),
            ("Senior Data Scientist", "OYO", "2017 - 2020",
             "Built dynamic pricing models for budget hotels."),
            ("Data Scientist", "Mu Sigma", "2015 - 2017",
             "Marketing mix modeling for CPG clients."),
        ],
        "education": [
            ("MBA", "IIM Calcutta", "2015"),
            ("B.Stat", "Indian Statistical Institute", "2013"),
        ],
        "projects": [
            "Co-author of an internal causal inference handbook.",
        ],
    },
    {
        "name": "Tara Joseph",
        "title": "Cloud Solutions Architect",
        "email": "tara.joseph@example.com",
        "phone": "+1-415-555-0142",
        "location": "San Francisco, USA",
        "years": 11,
        "summary": "Solutions architect with deep AWS and multi-cloud expertise.",
        "skills": [
            "AWS", "GCP", "Azure", "Kubernetes", "Terraform",
            "Serverless", "Microservices", "Java", "Python", "Solution Architecture",
        ],
        "experience": [
            ("Principal Solutions Architect", "Snowflake", "2020 - Present",
             "Leads enterprise migration engagements for the West region."),
            ("Senior SA", "AWS", "2015 - 2020",
             "Worked with Fortune 500 customers on EKS adoption."),
            ("Software Engineer", "Oracle", "2012 - 2015", "Java/EE backend services."),
        ],
        "education": [
            ("M.S., CS", "University of Texas Austin", "2012"),
            ("B.Tech, CSE", "NIT Surathkal", "2010"),
        ],
        "projects": [
            "AWS Solutions Architect Professional certified.",
        ],
    },
    {
        "name": "Daniel Park",
        "title": "Product Manager",
        "email": "daniel.park@example.com",
        "phone": "+1-650-555-0177",
        "location": "Mountain View, USA",
        "years": 8,
        "summary": "Product manager specializing in developer tools and APIs.",
        "skills": [
            "Product Management", "Roadmapping", "SQL", "A/B Testing",
            "User Research", "APIs", "Stakeholder Management",
        ],
        "experience": [
            ("Group PM", "Stripe", "2021 - Present",
             "Owned the Issuing API; grew adoption 4x in 2 years."),
            ("PM", "Twilio", "2017 - 2021",
             "Launched the Verify API."),
        ],
        "education": [
            ("MBA", "Stanford GSB", "2017"),
            ("B.S., EECS", "UC Berkeley", "2013"),
        ],
        "projects": ["Mentor at On Deck PM Fellowship."],
    },
    {
        "name": "Isha Bansal",
        "title": "QA Automation Engineer",
        "email": "isha.bansal@example.com",
        "phone": "+91-91111-22222",
        "location": "Noida, India",
        "years": 5,
        "summary": "QA engineer focused on test automation for web and mobile.",
        "skills": [
            "Selenium", "Playwright", "Cypress", "Java", "Python",
            "Appium", "TestRail", "JMeter", "API Testing",
        ],
        "experience": [
            ("Senior QA Engineer", "MakeMyTrip", "2021 - Present",
             "Built end-to-end Playwright suites for the booking funnel."),
            ("QA Engineer", "HCL", "2019 - 2021",
             "Manual + Selenium tests for banking clients."),
        ],
        "education": [
            ("B.Tech, IT", "Amity University", "2019"),
        ],
        "projects": ["ISTQB certified."],
    },
    {
        "name": "Mohammed Imran",
        "title": "Mobile Engineer (Android)",
        "email": "mohammed.imran@example.com",
        "phone": "+91-92222-33333",
        "location": "Bangalore, India",
        "years": 6,
        "summary": "Android engineer specializing in offline-first apps.",
        "skills": [
            "Kotlin", "Java", "Android", "Jetpack Compose", "Room",
            "Coroutines", "Firebase", "MVVM",
        ],
        "experience": [
            ("Senior Android Engineer", "Khatabook", "2021 - Present",
             "Owns the offline sync engine."),
            ("Android Engineer", "Times Internet", "2017 - 2021",
             "Worked on the TOI app."),
        ],
        "education": [
            ("B.E., CSE", "MS Ramaiah Institute of Technology", "2017"),
        ],
        "projects": ["Speaker at DroidCon India 2023."],
    },
    {
        "name": "Kavya Reddy",
        "title": "iOS Engineer",
        "email": "kavya.reddy@example.com",
        "phone": "+91-93333-44444",
        "location": "Hyderabad, India",
        "years": 4,
        "summary": "iOS engineer with consumer fintech experience.",
        "skills": [
            "Swift", "SwiftUI", "Objective-C", "iOS", "Combine",
            "Core Data", "XCTest",
        ],
        "experience": [
            ("iOS Engineer", "CRED", "2022 - Present",
             "Built the rewards module."),
            ("iOS Engineer", "Capillary", "2020 - 2022",
             "Worked on loyalty apps."),
        ],
        "education": [
            ("B.E., CSE", "Osmania University", "2020"),
        ],
        "projects": ["WWDC scholarship recipient 2019."],
    },
    {
        "name": "Liam Walsh",
        "title": "Site Reliability Engineer",
        "email": "liam.walsh@example.com",
        "phone": "+353-1-555-0123",
        "location": "Dublin, Ireland",
        "years": 9,
        "summary": "SRE focused on observability and incident response.",
        "skills": [
            "Kubernetes", "AWS", "Go", "Python", "Prometheus",
            "Grafana", "Loki", "Tempo", "Linux", "SLOs",
        ],
        "experience": [
            ("Staff SRE", "Stripe (Dublin)", "2020 - Present",
             "Owns the global observability platform."),
            ("SRE", "Intercom", "2016 - 2020",
             "Built incident management tooling."),
        ],
        "education": [
            ("B.Sc., Computer Science", "Trinity College Dublin", "2015"),
        ],
        "projects": ["Co-author of the internal incident postmortem template."],
    },
    {
        "name": "Olivia Brown",
        "title": "Marketing Analyst",
        "email": "olivia.brown@example.com",
        "phone": "+1-212-555-0190",
        "location": "New York, USA",
        "years": 3,
        "summary": "Marketing analyst skilled in funnel and attribution analysis.",
        "skills": [
            "SQL", "Excel", "Tableau", "Looker", "Python",
            "Google Analytics", "A/B Testing", "Attribution",
        ],
        "experience": [
            ("Marketing Analyst", "Warby Parker", "2022 - Present",
             "Owns paid social attribution dashboards."),
            ("Junior Analyst", "Sephora", "2021 - 2022",
             "Email campaign analytics."),
        ],
        "education": [
            ("B.A., Economics", "NYU", "2021"),
        ],
        "projects": [],
    },
    {
        "name": "Sai Krishna",
        "title": "NLP Engineer",
        "email": "sai.krishna@example.com",
        "phone": "+91-94444-66666",
        "location": "Bangalore, India",
        "years": 5,
        "summary": "NLP engineer focused on multilingual question answering for Indian languages.",
        "skills": [
            "Python", "PyTorch", "Hugging Face", "Transformers",
            "NLP", "RAG", "Multilingual", "Indic NLP", "FastAPI",
        ],
        "experience": [
            ("NLP Engineer", "AI4Bharat", "2021 - Present",
             "Trained Indic-BERT variants and built QA datasets in 11 languages."),
            ("ML Engineer", "Reverie Language Tech", "2019 - 2021",
             "Built transliteration models."),
        ],
        "education": [
            ("M.Tech, AI", "IIIT Bangalore", "2019"),
            ("B.Tech, CSE", "JNTU Hyderabad", "2017"),
        ],
        "projects": ["Co-author of IndicGLUE benchmark."],
    },
    {
        "name": "Nikhil Joshi",
        "title": "Cybersecurity Analyst",
        "email": "nikhil.joshi@example.com",
        "phone": "+91-95555-77777",
        "location": "Pune, India",
        "years": 7,
        "summary": "Security analyst specializing in cloud security and incident response.",
        "skills": [
            "AWS Security", "SIEM", "Splunk", "Python", "Threat Modeling",
            "Pentesting", "Burp Suite", "OWASP",
        ],
        "experience": [
            ("Senior Security Analyst", "TCS", "2020 - Present",
             "Lead investigator on a 2023 ransomware incident."),
            ("Security Analyst", "Wipro", "2017 - 2020",
             "Tier-2 SOC analyst."),
        ],
        "education": [
            ("B.Tech, IT", "MIT Pune", "2017"),
        ],
        "projects": ["CISSP certified."],
    },
    {
        "name": "Ritika Agarwal",
        "title": "Business Analyst",
        "email": "ritika.agarwal@example.com",
        "phone": "+91-96666-88888",
        "location": "Mumbai, India",
        "years": 4,
        "summary": "Business analyst at the intersection of finance and product.",
        "skills": [
            "SQL", "Excel", "Power BI", "Python", "Stakeholder Management",
            "Requirements Gathering", "Process Mapping",
        ],
        "experience": [
            ("Business Analyst", "Goldman Sachs", "2021 - Present",
             "Owns trade exception monitoring dashboards."),
            ("Junior BA", "Deloitte", "2020 - 2021",
             "Insurance domain projects."),
        ],
        "education": [
            ("MBA, Finance", "NMIMS Mumbai", "2020"),
            ("B.Com", "St. Xavier's Mumbai", "2018"),
        ],
        "projects": [],
    },
    {
        "name": "James O'Connor",
        "title": "Java Backend Engineer",
        "email": "james.oconnor@example.com",
        "phone": "+44-20-7946-0958",
        "location": "London, UK",
        "years": 10,
        "summary": "Java backend engineer for high-throughput trading systems.",
        "skills": [
            "Java", "Spring Boot", "Kafka", "PostgreSQL", "Redis",
            "Microservices", "Kubernetes", "AWS", "System Design",
        ],
        "experience": [
            ("Staff Engineer", "Revolut", "2020 - Present",
             "Owns the FX execution service."),
            ("Senior Engineer", "Goldman Sachs", "2015 - 2020",
             "Worked on GS-DB market data platform."),
            ("Software Engineer", "Thomson Reuters", "2013 - 2015", "Eikon plugins."),
        ],
        "education": [
            ("M.Sc., CS", "Imperial College London", "2013"),
            ("B.Sc., CS", "University of Manchester", "2012"),
        ],
        "projects": ["JCP member; contributor to OpenJDK."],
    },
    {
        "name": "Hina Kulkarni",
        "title": "Engineering Manager",
        "email": "hina.kulkarni@example.com",
        "phone": "+91-97777-99999",
        "location": "Bangalore, India",
        "years": 12,
        "summary": "EM with 12 years experience, last 4 leading platform teams of 8-15 engineers.",
        "skills": [
            "Engineering Management", "Hiring", "System Design",
            "Python", "Java", "AWS", "Kubernetes", "Coaching",
        ],
        "experience": [
            ("Engineering Manager", "Atlassian", "2020 - Present",
             "Manages two platform teams (12 engineers)."),
            ("Tech Lead", "ThoughtWorks", "2015 - 2020",
             "Tech lead on multiple consulting engagements."),
            ("Software Engineer", "Infosys", "2012 - 2015", "Java services."),
        ],
        "education": [
            ("B.Tech, CSE", "VJTI Mumbai", "2012"),
        ],
        "projects": ["Speaker at LeadDev London 2023."],
    },
    {
        "name": "Yash Malhotra",
        "title": "Frontend Engineer",
        "email": "yash.malhotra@example.com",
        "phone": "+91-98888-11111",
        "location": "Delhi, India",
        "years": 3,
        "summary": "Frontend engineer comfortable with React and design systems.",
        "skills": [
            "JavaScript", "TypeScript", "React", "Next.js", "Tailwind CSS",
            "Storybook", "Webpack", "Accessibility",
        ],
        "experience": [
            ("Frontend Engineer", "Hashnode", "2022 - Present",
             "Owns the editor surface."),
            ("Junior FE Engineer", "Unacademy", "2021 - 2022",
             "Worked on the live class viewer."),
        ],
        "education": [
            ("B.Tech, ECE", "NSUT Delhi", "2021"),
        ],
        "projects": ["Maintainer of a React DataGrid library (700 stars)."],
    },
    {
        "name": "Pooja Desai",
        "title": "Bioinformatics Researcher",
        "email": "pooja.desai@example.com",
        "phone": "+91-99999-22222",
        "location": "Bangalore, India",
        "years": 6,
        "summary": "Researcher applying ML to genomic data.",
        "skills": [
            "Python", "R", "Machine Learning", "Bioinformatics",
            "Genomics", "Bioconductor", "PyTorch", "Statistics",
        ],
        "experience": [
            ("Senior Researcher", "Strand Life Sciences", "2020 - Present",
             "ML for variant classification."),
            ("Research Associate", "NCBS Bangalore", "2018 - 2020",
             "Single-cell RNA-seq pipelines."),
        ],
        "education": [
            ("Ph.D., Computational Biology", "IISc Bangalore", "2018"),
            ("M.Sc., Biotech", "University of Pune", "2013"),
        ],
        "projects": ["Co-author on Nature Communications 2022."],
    },
    {
        "name": "Sandeep Yadav",
        "title": "Embedded Systems Engineer",
        "email": "sandeep.yadav@example.com",
        "phone": "+91-90000-33333",
        "location": "Bangalore, India",
        "years": 8,
        "summary": "Embedded engineer for automotive ECUs.",
        "skills": [
            "C", "C++", "RTOS", "ARM Cortex", "AUTOSAR",
            "CAN", "MISRA C", "Embedded Linux",
        ],
        "experience": [
            ("Senior Embedded Engineer", "Bosch India", "2018 - Present",
             "Owns powertrain ECU firmware."),
            ("Embedded Engineer", "Continental", "2015 - 2018",
             "Sensor fusion firmware."),
        ],
        "education": [
            ("M.Tech, VLSI", "IIT Kharagpur", "2015"),
            ("B.E., ECE", "BMS College of Engineering", "2013"),
        ],
        "projects": [],
    },
    {
        "name": "Riya Choudhary",
        "title": "Recruiter (Tech)",
        "email": "riya.choudhary@example.com",
        "phone": "+91-91111-44444",
        "location": "Bangalore, India",
        "years": 5,
        "summary": "Tech recruiter with focus on senior IC and EM hiring.",
        "skills": [
            "Sourcing", "LinkedIn Recruiter", "ATS", "Boolean Search",
            "Tech Hiring", "Negotiation",
        ],
        "experience": [
            ("Senior Tech Recruiter", "Razorpay", "2021 - Present",
             "Hired 70+ engineers in 2 years."),
            ("Tech Recruiter", "Naukri.com", "2019 - 2021",
             "Recruitment partner for SaaS clients."),
        ],
        "education": [
            ("MBA, HR", "XLRI Jamshedpur", "2019"),
            ("B.A., Psychology", "Lady Shri Ram College", "2017"),
        ],
        "projects": [],
    },
    {
        "name": "Aditya Rao",
        "title": "GenAI Engineer",
        "email": "aditya.rao@example.com",
        "phone": "+91-93333-55555",
        "location": "Bangalore, India",
        "years": 4,
        "summary": (
            "Applied GenAI engineer building RAG and agentic systems on top of LLMs."
        ),
        "skills": [
            "Python", "LangChain", "LlamaIndex", "RAG", "Vector Databases",
            "ChromaDB", "Pinecone", "OpenAI", "LLMs", "Prompt Engineering",
            "Machine Learning",
        ],
        "experience": [
            ("Senior ML Engineer", "Sarvam AI", "2023 - Present",
             "Built a multi-agent RAG system over enterprise PDFs; cut ticket "
             "deflection time from 8 minutes to 90 seconds."),
            ("ML Engineer", "Glean", "2021 - 2023",
             "Worked on enterprise search ranking."),
        ],
        "education": [
            ("B.Tech, CSE", "BITS Goa", "2021"),
        ],
        "projects": [
            "Built `chroma-rag-toolkit` (open source) used by 3K+ developers.",
        ],
    },
    {
        "name": "Megha Pillai",
        "title": "Senior Python Backend Engineer",
        "email": "megha.pillai@example.com",
        "phone": "+91-94444-66666",
        "location": "Bangalore, India",
        "years": 6,
        "summary": (
            "Backend engineer with 6 years of Python experience; recently shipped "
            "a fully async FastAPI platform handling 30K RPS."
        ),
        "skills": [
            "Python", "FastAPI", "asyncio", "PostgreSQL", "Redis",
            "Kafka", "Docker", "Kubernetes", "AWS", "System Design",
            "Machine Learning",
        ],
        "experience": [
            ("Senior Backend Engineer", "Razorpay", "2021 - Present",
             "Owns the disputes service. Migrated from Django to FastAPI; "
             "P99 latency dropped from 320ms to 90ms."),
            ("Backend Engineer", "Hotstar", "2019 - 2021",
             "Worked on the personalization API."),
        ],
        "education": [
            ("B.Tech, CSE", "NIT Calicut", "2019"),
        ],
        "projects": [
            "Speaker at PyCon India 2023 on async patterns.",
        ],
    },
    {
        "name": "Lakshmi Narayanan",
        "title": "Lead Data Scientist",
        "email": "lakshmi.n@example.com",
        "phone": "+91-95555-77777",
        "location": "Chennai, India",
        "years": 11,
        "summary": (
            "Lead DS with 11 years across e-commerce and travel. Built ML platforms "
            "from 0->1 twice."
        ),
        "skills": [
            "Python", "SQL", "Spark", "Machine Learning", "Deep Learning",
            "MLOps", "AWS", "Kubernetes", "Mentorship",
        ],
        "experience": [
            ("Lead Data Scientist", "MakeMyTrip", "2020 - Present",
             "Owns the demand forecasting suite (hotel + flights)."),
            ("Senior Data Scientist", "Amazon India", "2016 - 2020",
             "Built last-mile ETA models."),
            ("Data Scientist", "Mu Sigma", "2013 - 2016",
             "Marketing analytics for FMCG."),
        ],
        "education": [
            ("M.S., Stats", "University of Michigan", "2013"),
            ("B.Tech, CSE", "Anna University", "2011"),
        ],
        "projects": ["Mentor on TopMate; 200+ sessions."],
    },
    {
        "name": "Raghav Bhalla",
        "title": "Junior Software Engineer",
        "email": "raghav.bhalla@example.com",
        "phone": "+91-96666-99999",
        "location": "Bangalore, India",
        "years": 1,
        "summary": "Recent grad, 1 year of professional Python and React experience.",
        "skills": [
            "Python", "JavaScript", "React", "Flask", "SQL",
            "Git", "Docker",
        ],
        "experience": [
            ("Software Engineer", "Zoho", "2024 - Present",
             "Builds internal admin tooling in Flask + React."),
            ("Software Intern", "Zoho", "2023 - 2024",
             "Bug fixing on the CRM."),
        ],
        "education": [
            ("B.E., CSE", "PES University", "2024"),
        ],
        "projects": ["Personal RAG-on-PDFs side project (GitHub)."],
    },
    {
        "name": "Emily Chen",
        "title": "Senior Python ML Engineer",
        "email": "emily.chen@example.com",
        "phone": "+1-415-555-2233",
        "location": "Seattle, USA",
        "years": 7,
        "summary": (
            "Senior ML engineer with 7 years of Python and a strong production-ML "
            "track record. Loves clean APIs."
        ),
        "skills": [
            "Python", "PyTorch", "scikit-learn", "MLflow", "Airflow",
            "Spark", "AWS", "Kubernetes", "Machine Learning", "Deep Learning",
            "MLOps",
        ],
        "experience": [
            ("Senior ML Engineer", "Stripe", "2021 - Present",
             "Owns fraud model lifecycle: training, serving, monitoring."),
            ("ML Engineer", "Lyft", "2018 - 2021",
             "ETA models for the marketplace team."),
        ],
        "education": [
            ("M.S., CS", "University of Washington", "2018"),
            ("B.S., CS", "UC San Diego", "2016"),
        ],
        "projects": [
            "Open-source: ml-monitoring-toolkit (1.8K stars).",
        ],
    },
    {
        "name": "Farhan Ahmed",
        "title": "Senior Data Engineer (Streaming)",
        "email": "farhan.ahmed@example.com",
        "phone": "+91-97777-11111",
        "location": "Hyderabad, India",
        "years": 8,
        "summary": "Streaming-first data engineer; production Kafka, Flink and Spark.",
        "skills": [
            "Python", "Scala", "Kafka", "Flink", "Spark Streaming",
            "Airflow", "AWS", "Snowflake", "SQL", "dbt",
        ],
        "experience": [
            ("Senior Data Engineer", "Uber India", "2020 - Present",
             "Owns the surge-pricing event pipeline."),
            ("Data Engineer", "Walmart Labs", "2016 - 2020",
             "Built Spark jobs for inventory."),
        ],
        "education": [
            ("B.Tech, CSE", "IIT Guwahati", "2016"),
        ],
        "projects": ["Apache Flink contributor (3 PRs merged)."],
    },
]


# ---------------------------------------------------------------------------
# Job Description templates
# ---------------------------------------------------------------------------

JOBS: list[dict] = [
    {
        "id": "01_senior_python_ml_engineer",
        "title": "Senior Python ML Engineer",
        "company": "Acme AI",
        "location": "Bangalore / Remote (India)",
        "summary": (
            "We are looking for a Senior Python ML Engineer with 5+ years of "
            "experience to own end-to-end model lifecycles, from data to "
            "production serving, on AWS."
        ),
        "must_have": [
            "5+ years of professional Python experience",
            "Hands-on Machine Learning / Deep Learning in production",
            "Production experience with AWS or GCP",
            "Comfort with Docker and Kubernetes",
        ],
        "nice_to_have": [
            "PyTorch", "MLOps tooling (MLflow, Airflow, Kubeflow)",
            "Experience leading or mentoring engineers",
        ],
        "responsibilities": [
            "Design, train and ship ML models for ranking and recommendations",
            "Own CI/CD and monitoring for the model serving stack",
            "Partner with platform and product teams on data pipelines",
        ],
    },
    {
        "id": "02_genai_rag_engineer",
        "title": "GenAI / RAG Engineer",
        "company": "NovaSearch",
        "location": "Remote (India)",
        "summary": (
            "Build retrieval-augmented generation systems on top of large "
            "language models. You will own the retrieval, ranking and prompt "
            "stack of our flagship enterprise search product."
        ),
        "must_have": [
            "3+ years of Python in production",
            "Hands-on experience with LLMs (OpenAI / Llama / Mistral)",
            "Vector databases (ChromaDB, Pinecone, Weaviate or FAISS)",
            "Strong NLP fundamentals",
        ],
        "nice_to_have": [
            "LangChain / LlamaIndex",
            "Prompt engineering and evaluation harnesses",
            "Multilingual NLP",
        ],
        "responsibilities": [
            "Design chunking and embedding strategies for enterprise PDFs",
            "Implement hybrid retrieval (semantic + BM25)",
            "Build evaluation pipelines for retrieval quality and hallucinations",
        ],
    },
    {
        "id": "03_senior_data_engineer_streaming",
        "title": "Senior Data Engineer - Streaming",
        "company": "Trailblazer Logistics",
        "location": "Hyderabad, India",
        "summary": (
            "Own the real-time data backbone of our logistics platform: Kafka, "
            "Flink and Spark Streaming."
        ),
        "must_have": [
            "6+ years of data engineering experience",
            "Production Kafka and either Flink or Spark Streaming",
            "Strong Python or Scala",
            "AWS",
        ],
        "nice_to_have": [
            "Snowflake / Databricks",
            "dbt",
        ],
        "responsibilities": [
            "Own ingestion and processing of 50K+ events/second",
            "Define SLOs for data freshness and correctness",
            "Mentor 2-3 mid-level engineers",
        ],
    },
    {
        "id": "04_engineering_manager_platform",
        "title": "Engineering Manager - Platform",
        "company": "Bluepine Technologies",
        "location": "Bangalore, India",
        "summary": (
            "Lead a 10-12 person platform team responsible for our internal "
            "developer platform (Kubernetes, CI/CD, observability)."
        ),
        "must_have": [
            "8+ years of engineering experience, 3+ years managing engineers",
            "Strong systems / backend background (Python or Java)",
            "Experience with Kubernetes and AWS in production",
            "Track record of hiring and growing engineers",
        ],
        "nice_to_have": [
            "Worked on internal developer platforms previously",
            "Open-source contributions",
        ],
        "responsibilities": [
            "Own roadmap and quarterly planning for the platform team",
            "Hire 4 engineers in the next 2 quarters",
            "Coach senior engineers and tech leads",
        ],
    },
    {
        "id": "05_full_stack_developer",
        "title": "Full Stack Developer (React + Node)",
        "company": "BrightCart",
        "location": "Gurgaon, India",
        "summary": (
            "Build customer-facing features end-to-end across a React frontend "
            "and Node.js BFF for our e-commerce checkout."
        ),
        "must_have": [
            "3+ years of React in production",
            "Node.js / Express experience",
            "TypeScript",
            "Comfort with relational databases (PostgreSQL or MySQL)",
        ],
        "nice_to_have": [
            "Next.js",
            "AWS / Docker",
            "Experience with checkout / payments",
        ],
        "responsibilities": [
            "Own end-to-end delivery of checkout features",
            "Improve frontend Web Vitals",
            "Collaborate with designers and PMs in a small squad",
        ],
    },
    {
        "id": "06_computer_vision_engineer",
        "title": "Computer Vision Engineer",
        "company": "AutonomyLabs",
        "location": "Bangalore, India",
        "summary": (
            "Design and ship perception models (detection, tracking, "
            "segmentation) for our autonomous industrial vehicles."
        ),
        "must_have": [
            "5+ years in Computer Vision / Deep Learning",
            "Strong Python and C++",
            "PyTorch or TensorFlow",
            "Experience deploying models on edge (TensorRT or similar)",
        ],
        "nice_to_have": [
            "ROS / ROS2",
            "CUDA",
            "Experience in robotics or autonomous vehicles",
        ],
        "responsibilities": [
            "Own the perception stack roadmap",
            "Train and deploy detection / tracking models",
            "Optimize inference for embedded GPUs",
        ],
    },
]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_resume(r: dict) -> str:
    parts: list[str] = []
    parts.append(f"{r['name']}")
    parts.append(f"{r['title']}")
    parts.append(f"Email: {r['email']} | Phone: {r['phone']} | Location: {r['location']}")
    parts.append(f"Total Experience: {r['years']} years")
    parts.append("")
    parts.append("SUMMARY")
    parts.append(r["summary"])
    parts.append("")
    parts.append("SKILLS")
    parts.append(", ".join(r["skills"]))
    parts.append("")
    parts.append("EXPERIENCE")
    for title, company, dates, desc in r["experience"]:
        parts.append(f"- {title}, {company} ({dates})")
        parts.append(f"  {desc}")
    parts.append("")
    parts.append("EDUCATION")
    for degree, school, year in r["education"]:
        parts.append(f"- {degree}, {school} ({year})")
    if r["projects"]:
        parts.append("")
        parts.append("PROJECTS")
        for p in r["projects"]:
            parts.append(f"- {p}")
    return "\n".join(parts) + "\n"


def render_jd(j: dict) -> str:
    parts: list[str] = []
    parts.append(f"{j['title']}")
    parts.append(f"{j['company']} | {j['location']}")
    parts.append("")
    parts.append("ABOUT THE ROLE")
    parts.append(j["summary"])
    parts.append("")
    parts.append("MUST HAVE")
    for m in j["must_have"]:
        parts.append(f"- {m}")
    parts.append("")
    parts.append("NICE TO HAVE")
    for m in j["nice_to_have"]:
        parts.append(f"- {m}")
    parts.append("")
    parts.append("RESPONSIBILITIES")
    for m in j["responsibilities"]:
        parts.append(f"- {m}")
    return "\n".join(parts) + "\n"


def slugify(name: str) -> str:
    return name.lower().replace(" ", "_").replace("'", "").replace(".", "")


def main() -> None:
    for r in RESUMES:
        path = RESUMES_DIR / f"{slugify(r['name'])}.txt"
        path.write_text(render_resume(r), encoding="utf-8")

    for j in JOBS:
        path = JD_DIR / f"{j['id']}.txt"
        path.write_text(render_jd(j), encoding="utf-8")

    print(f"Wrote {len(RESUMES)} resumes -> {RESUMES_DIR}")
    print(f"Wrote {len(JOBS)} job descriptions -> {JD_DIR}")


if __name__ == "__main__":
    main()
