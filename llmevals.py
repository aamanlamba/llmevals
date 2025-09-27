# LLM Eval experiments
# from Taming LLMs - https://tamingllm.substack.com/
from dotenv import load_dotenv
import os
import logging
logging.basicConfig(level=logging.INFO)

load_dotenv()
from openai import OpenAI
import pandas as pd

def generate_responses (
        model_name: str,
        prompt: str,
        temperatures: list[float],
        attempts: int = 3
    ) -> pd.DataFrame:
    """
    Generate responses from an LLM for a given prompt at different temperatures.
    Args:
        model_name (str): The name of the LLM model to use.
        prompt (str): The prompt to send to the LLM.
        temperatures (list[float]): A list of temperature values to use for generation.
        attempts (int): Number of attempts to generate a response for each temperature.
    Returns:
        pd.DataFrame: A DataFrame containing the model name, temperature, and generated responses.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    results = []

    for temp in temperatures:
        for attempt in range(attempts):
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temp,
                max_tokens=50
            )
            
            results.append({
                'temperature': temp,
                'attempt': attempt + 1,
                'response': response.choices[0].message.content
            })

            # Display results grouped by temperature
            df_results = pd.DataFrame(results)
            for temp in temperatures:
                logging.log(f"\nTemperature = {temp}")
                logging.log("-" * 40)
                temp_responses = df_results[df_results['temperature'] == temp]
                for _, row in temp_responses.iterrows():
                    logging.log(f"Attempt {row['attempt']}: {row['response']}")
            
            return df_results
        
if __name__ == "__main__":
    model = "gpt-3.5-turbo"
    prompt = "List three creative uses for a paperclip."
    temperatures = [0.0, 0.5, 1.0]
    
    df = generate_responses(model, prompt, temperatures)
    df.to_csv("llm_responses.csv", index=False)
    logging.log("\nResponses saved to llm_responses.csv")