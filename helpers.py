from nltk.tokenize import sent_tokenize, word_tokenize
import unicodedata
from nltk.tokenize.treebank import TreebankWordDetokenizer
# https://stackoverflow.com/questions/56856394/rejoin-sentence-like-original-after-tokenizing-with-nltk-word-tokenize

def split_doc(text, max_length):
    """
    doc: document >=4096 word
    max_length: length maximun to input summary
    output: split doc to mul small doc
    """

    word_tokens_split = word_tokenize(text)
    if len(word_tokens_split)<=max_length:
        return [text]
    sentence_tokens = sent_tokenize(text)

    length_text = len(word_tokens_split)
    doc_split_num = length_text // max_length + 1

    word_in_sub_doc = max_length // doc_split_num

    count = 0
    text  = ""
    result = []
    for sentence in sentence_tokens:
        count += len(word_tokenize(sentence))
        if count > word_in_sub_doc:
            count = 0
            result.append(text)
            text = ""
        text += " ".join(sentence.split()) +" "
    
    return result



if __name__ =="__main__":
    text_ip ="""
    In early March, after two ultra-fast delivery startups shut down in New York City in a single week, a self-proclaimed pioneer in the space appeared to see an opportunity for some media attention.
    Getir, a Turkish startup founded in 2015, had recently raised $768 million in funding valuing it at $11.8 billion, "cementing its position as a decacorn" even in the face of a "volatile" market, according to a representative for the company at the time. The representative suggested a reporter discuss with Getir's CEO the future of the industry amid the "disappearance" of two smaller competitors.
    Two months later, Getir slashed 14% of its global workforce, or nearly 4,500 employees, according to multiple reports at the time.
    The sudden shift reflected the broader turbulence in this sector. Buoyed by billions in venture capital and a surge in demand earlier in the pandemic, a long list of on-demand companies promised to deliver ice cream, toilet paper, vodka or even a single apple in as little as 10 minutes. These startups opened offices and micro-fulfillment centers in cities across the country and rapidly recruited couriers.
    Then the music stopped. Soaring inflation, rising interest rates, fears of a looming recession and a war in Ukraine forced much of the tech industry to rethink expenses. Perhaps nowhere was that pullback as severe as this flashy, and very costly, corner of the on-demand industry — an industry which was born out of the Great Recession of 2008 and had never experienced a prolonged downturn.
    "It's the model we saw with Uber a decade ago of heavily prioritizing growth over profits to be able to rapidly seize first-mover advantage," said Alex Frederick, a senior analyst focused on emerging technology at PitchBook, a data analytics company. This model "requires high burn, high capital investments to continually expand into new markets, attract customers and retain them," he said. Now, investors may have less appetite for it.
    This year, Fridge No More and Buyk have shut down operations completely; Jokr said it would shut down its US operations and hone in on its Latin America business; and Gopuff, Gorillas and Getir have each had at least one round of layoffs. Combined, at least 8,250 jobs have been lost, according to a tally by CNN Business based on a combination of press coverage, publicly available information and confirmations from some of the companies. Many of those impacted worked as couriers, or the frontline workers essential to delivering on the missions of speed and convenience.
    The fallout has created whiplash for some of the many workers who bet on it. One of the thousands of employees laid off by Getir told CNN Business that they had felt a sense of security because of something they said they'd heard expressed inside the seven-year-old company: It had never undergone layoffs.
    The worker, who joined in late 2021 as Getir was building up its US presence in Boston and Chicago, said that while they'd rationally understood it's common for startups to lay off employees, or even to fail, they believed Getir would be the exception. "I really believed they just didn't do layoffs and thought they were going to be a very different startup," said the former employee, who asked their name to be withheld for fear of retaliation. "I believed the dream they were selling. I'm disappointed."
    In an e-mailed response to CNN Business, Getir CEO Nazim Salur said his company "decided to extend the runway" through layoffs "because of the deterioration of market conditions." Salur added: "Layoffs are something we try not to do unless it is absolutely necessary. This is the first time in Getir's seven-year history that Getir has gone through a workforce cut of this magnitude."
    He declined to provide additional details on the round of cuts in May, including what portion of its US business was impacted. The company said it's the first instance of widespread layoffs as opposed to specific store closures.
    Like Getir, the ultra-fast delivery startups that remain have largely indicated the cuts are intended to help them weather the economic downturn. As they adjust, however, other businesses appear to be eying opportunities to gain a footing in the market, and perhaps change how it operates.
    The rise and stumbles of 15-minute delivery startups
    When Gopuff, considered the first to leverage its own stores to fulfill ultra-fast deliveries, was founded in 2013, Uber and the app-based gig economy were only a few years old. Gopuff's original pitch was to offer hookah deliveries and later food to college students with sudden cravings.
    Less than a decade later, Gopuff was valued at $15 billion, and then reportedly raised a convertible note at a valuation cap of up to $40 billion. It was also operating in more than 1,200 cities and capturing the attention of a more established gig company, Uber, in the form of a partnership. Gopuff had 450 micro-fulfillment centers spread across college towns and major cities to supply its customers with everything from food to alcohol and medicine almost instantaneously. As of March 2022, it had 15,000 employees.
    But in a memo to investors in July, it outlined a number of changes it was making, including a second round of layoffs in a matter of months and closing 76 micro-fulfillment centers, to prepare for the next two years by which time it projects it can be profitable. It said that it is preparing for "what could be a much more significant macro-economic downturn than we are experiencing currently."
    "The instant commerce industry that Gopuff created is at an inflection point," the company said in the memo, a copy of which was viewed by CNN Business. "Gopuff was among the last hyper-growth tech companies to raise a significant round in the previous economic environment and among the first to cut costs to focus on and optimize unit economics," the company said, referring to the revenue and costs associated with each delivery.
    Even with the cuts at Gopuff and other startups, some industry watchers have doubts about the long-term viability of their main offerings, especially in a more sober economic environment.
    Brittain Ladd, a supply chain consultant who has advised a number of companies in the space and previously worked in strategy at Amazon, told CNN Business that the premise of 15-minute delivery is a "gimmick."
    "It hooked people, it generated a lot of publicity, it became something of an oddity," he said. "The goal was to get consumers to ask, 'Why is it I can get my groceries in 15 minutes, but I can't get cosmetics and shoes and apparel and things like that [in 15 minutes]?'"
    "That's where the next phase of growth was going to be," he added. Then came the downturn and fears of a recession. "Investors collectively said, 'How in the world are we ever going to make money investing in this?'"
    A slower path forward for some
    As some of the biggest names in the ultra-fast delivery sector stumble, others are looking to gain ground.
    Instacart, which in May filed paperwork to go public, debuted an ultra-fast delivery offering for certain customers of Publix in Miami. Instacart declined to share metrics on the partnership, but a company spokesperson said it has seen interest in the Publix offering, which is called Publix Quick Picks. A Publix spokesperson did not respond to a request for comment.
    In December, DoorDash began offering an ultra-fast delivery option in New York City from a DashMart, one of the stores it opened in 2020, and has since begun expanding.
    Meanwhile, more under-the-radar companies haven taken what they say are intentionally slower and more methodical approaches to ultra-fast delivery. Paul Stellatos, who has been running grocery stores in the Chicago area for two decades, told CNN Business his company, Go Grocer, had been developing an app to offer quick deliveries from its stores -- without VC-backing. That process took many months, during which time Gorillas and Getir were taking the market by storm.
    "We were just kind of sitting back and saying, 'Okay, Mr. Gorillas or Mr. Getir, how are you planning on making money capturing the audience and getting a sticky customer, somebody who is gonna order more than once?" said Stellatos, who said that Go Grocer leverages workers from companies like DoorDash and Uber to deliver orders from its stores. "I'm really a stickler on money because we don't have the opportunity of burning cash. We can only be profitable."
    Vitaly Alexandrov, the CEO and founder of San Francisco-based startup Food Rocket, which offers 10- to 30-minute grocery deliveries with an emphasis on fresh food, said it has just six retail locations in San Francisco and Chicago to date.
    "It's not because we're a super slow company," said Alexandrov, who noted Food Rocket grew out of the remnants of his earlier business focused on restaurants that shuttered due to the pandemic. "It's because we tried to build a really sustainable business model. It couldn't be scalable without such huge losses."
    Alexandrov said the extra time of 30-minute deliveries allows for more "sustainable unit economics" because it provides more opportunities to group several deliveries for a single worker. Food Rocket, which has raised $30 million to date, is hoping to raise a new round of funding by the end of the year. "We still believe that people love to get everything super-fast," added Alexandrov.
    One of the most valuable companies in the world apparently thinks so, too. Amazon recently began testing drone deliveries in one town. The goal: fulfilling orders in 30 minutes.
    """

    result = split_doc(text_ip, 900)
    for r in result:
        print("*"*30)
        print(r)
