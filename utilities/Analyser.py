from groq import Groq
import json
import ast
import os
from dotenv import load_dotenv
load_dotenv()

def extract_and_parse(text: str):
    start = text.find('{')
    end = text.rfind('}')
    
    if start == -1 or end == -1 or start > end:
        return None
    
    json_str = text[start:end + 1]

    # Try strict JSON first
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    # Fallback: Python dict parsing
    try:
        return ast.literal_eval(json_str)
    except Exception:
        return None


def run_analysis(data):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    extra_instructions = ""
    completion = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "user",
                "content": f"""You are an expert Process Engineer in a Cement Factory, which most efficiently can analyse the data of CEMS and can tell what are can be the issues in the factory lleading to the values of pollutants.
                I will provide you with the future predicted data of CEMS and some extra instructions , you will analyse the data also keeping in mind the extra instructions provded to you.

                ##INPUT
                Predicted Data Frame : {data}
                Extra Instructions : {extra_instructions}

                ##OUTPUT
                Provide Output in the given below JSON Structure only strictly. No extra Information needed just this json structure.
                {{
                'Suggestions':[List of 5 suggestions as a list of strings],
                'Immediate Actionables' : [List if any immediate emergency action needed to be taken, else can be returned empty if no emergency and immediate action needed]
                }}
                """
            }
        ],
        temperature=0.01
    )


    return extract_and_parse(completion.choices[0].message.content)