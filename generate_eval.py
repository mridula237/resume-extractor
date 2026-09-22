import json
import os
import random
from openai import OpenAI

gpt = OpenAI()

NAMES = [
    "Alex Johnson", "Priya Patel", "Marcus Williams", "Sarah Chen", "James O'Brien",
    "Fatima Al-Hassan", "David Kim", "Emma Rodriguez", "Raj Sharma", "Lisa Thompson",
    "Michael Brown", "Aisha Okonkwo", "Carlos Mendez", "Yuki Tanaka", "Grace Liu",
    "Omar Abdullah", "Sophie Martin", "Arjun Nair", "Hannah Schmidt", "Kevin Park",
    "Isabella Rossi", "Daniel Osei", "Mei Wong", "Tyler Anderson", "Amara Diallo",
    "Nathan Clarke", "Zara Ahmed", "Lucas Dubois", "Nadia Petrov", "Sean Murphy"
]

ROLES = [
    "Data Scientist", "Data Engineer", "ML Engineer", "Analytics Engineer",
    "Software Engineer", "Product Manager", "Business Analyst", "Data Analyst",
    "DevOps Engineer", "Backend Engineer", "Frontend Engineer", "Full Stack Engineer",
    "Research Scientist", "AI Engineer", "Cloud Engineer", "Platform Engineer",
    "Security Engineer", "QA Engineer", "Marketing Analyst", "Financial Analyst"
]

COMPANIES = [
    "Google", "Meta", "Amazon", "Microsoft", "Apple", "Netflix", "Uber", "Airbnb",
    "Stripe", "Databricks", "Snowflake", "Palantir", "OpenAI", "Nvidia", "Salesforce",
    "Oracle", "IBM", "Accenture", "Deloitte", "McKinsey", "JPMorgan", "Goldman Sachs",
    "Tesla", "SpaceX", "Lyft", "Twitter", "LinkedIn", "Shopify", "Atlassian", "Twilio"
]

SCHOOLS = [
    "MIT", "Stanford University", "Carnegie Mellon University", "UC Berkeley",
    "University of Michigan", "Georgia Tech", "Cornell University", "UCLA",
    "University of Texas at Austin", "University of Washington", "NYU", "Columbia University",
    "Purdue University", "University of Illinois", "Ohio State University"
]

DEGREES = [
    "B.S. in Computer Science", "M.S. in Data Science", "B.S. in Statistics",
    "M.S. in Computer Science", "B.S. in Mathematics", "MBA",
    "M.S. in Machine Learning", "B.S. in Electrical Engineering",
    "M.S. in Information Systems", "B.S. in Software Engineering"
]

CITIES = [
    "San Francisco, CA", "New York, NY", "Seattle, WA", "Austin, TX",
    "Boston, MA", "Chicago, IL", "Los Angeles, CA", "Denver, CO",
    "Atlanta, GA", "Portland, OR", "Miami, FL", "Washington, DC"
]


def make_resume_prompt(name, role, companies, schools):
    return f"""Write a realistic resume for {name}, a {role} with 3-6 years of experience.

Use these details:
- Worked at: {', '.join(companies)}
- Studied at: {', '.join(schools)}
- Location: {random.choice(CITIES)}
- Include: email, phone, 2 education entries, 2-3 jobs, 10-15 skills, 1-2 certifications

Format it as plain text (no markdown), like a real resume with sections:
CONTACT, SUMMARY, EXPERIENCE, EDUCATION, SKILLS, CERTIFICATIONS

Be specific with dates, job titles, and bullet points. Make it realistic."""


def generate_resume_text(name, role):
    companies = random.sample(COMPANIES, 2)
    schools = random.sample(SCHOOLS, 2)
    prompt = make_resume_prompt(name, role, companies, schools)

    r = gpt.chat.completions.create(
        model="gpt-5.4-nano",
        max_completion_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    return r.choices[0].message.content


def generate_gold_json(resume_text):
    from schema import Resume

    r = gpt.beta.chat.completions.parse(
        model="gpt-5.4-nano",
        max_completion_tokens=1500,
        messages=[
            {"role": "system", "content": "Extract structured data from this resume. Be accurate and complete."},
            {"role": "user", "content": resume_text}
        ],
        response_format=Resume,
    )
    return json.loads(r.choices[0].message.content)


def main():
    os.makedirs("eval/resumes", exist_ok=True)
    os.makedirs("eval/gold", exist_ok=True)

    for i, (name, role) in enumerate(zip(NAMES, ROLES)):
        print(f"[{i+1}/30] Generating: {name} — {role}")

        # generate resume text
        text = generate_resume_text(name, role)
        resume_path = f"eval/resumes/{i+1:02d}_{name.replace(' ', '_')}.txt"
        with open(resume_path, "w") as f:
            f.write(text)

        # generate gold JSON
        gold = generate_gold_json(text)
        gold_path = f"eval/gold/{i+1:02d}_{name.replace(' ', '_')}.json"
        with open(gold_path, "w") as f:
            json.dump(gold, f, indent=2)

        print(f"   saved: {resume_path}")
        print(f"   saved: {gold_path}")

    print("\nDone. 30 resumes + gold JSON saved in eval/")


if __name__ == "__main__":
    main()