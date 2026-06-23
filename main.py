from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


def main():
    information = """
    Muhammad ibn Musa al-Khwarizmi was a mathematician Muhammad ibn Musa al-Khwarizmi,[note 1] or simply al-Khwarizmi (c. 780 – c. 850) was a mathematician active during the Islamic Golden Age, who produced Arabic-language works in mathematics, astronomy, and geography. Around 820, he worked at the House of Wisdom in Baghdad, the contemporary capital city of the Abbasid Caliphate. One of the most prominent scholars of the period, his works were widely influential on later authors, both in the Islamic world and Europe.

    Few biographical details are known about al-Khwarizmi's life. His popularizing treatise on algebra, compiled between 813 and 833 as Al-Jabr (The Compendious Book on Calculation by Completion and Balancing), presented the first systematic solution of linear and quadratic equations. One of his achievements in algebra was his demonstration of how to solve quadratic equations by completing the square, for which he provided geometric justifications. Because al-Khwarizmi was the first person to treat algebra as an independent discipline and introduced the methods of "reduction" and "balancing" (the transposition of subtracted terms to the other side of an equation, that is, the cancellation of like terms on opposite sides of the equation), he has been described as the father or founder of algebra. The English term algebra comes from the short-hand title of his aforementioned treatise (Al-Jabr, meaning "completion" or "rejoining").

    His name gave rise to the English terms algorism and algorithm. In the 12th century, Latin translations of al-Khwarizmi's textbook on Indian arithmetic introduced the decimal-based positional number system to the Western world. Likewise, Al-Jabr was translated into Latin and used for centuries as a principal mathematical textbook in European universities.

    Al-Khwarizmi revised Ptolemy's Geography, listing the longitudes and latitudes of cities and localities. He further produced astronomical tables and wrote about calendric works, the astrolabe, and the sundial. He also made important contributions to trigonometry, producing accurate sine and cosine tables.

    Al-Khwarizmi's name was latinized as Algoritmi, making his name the origin of the word "algorithm."
    """

    summary_template = """
    Given the information below about a person:

    {information}

    Create:
    1. A short summary.
    2. Two interesting facts about the person and their location.

    Format the response clearly.
    """

    prompt = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
    )

    chain = prompt | llm

    try:
        print("Generating response...\n")

        response = chain.invoke(
            {"information": information}
        )

        print(response.content)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()