

# import torch
from transformers import BigBirdPegasusForConditionalGeneration, AutoTokenizer
import time
import re 
import nltk
from nltk import tokenize
nltk.download('punkt')
#nltk.download('words')
words = set(nltk.corpus.words.words())
import helpers
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def replace_unneeded_items(string):
    replace_list = {
        # chuẩn hóa dấu câu
        # loại bỏ các dấu
        '\n':' ','"':'','‘':'','’':'','*':'', '(':'', ')':'','…':''
    }
    for k, v in replace_list.items():
        string = string.replace(k, v)

    string = re.sub("([.]{2,})","", string)
    string = re.sub(" +"," ", string)
    string = string.strip()
    return string

def remove_duplicate(x): 
  return list(dict.fromkeys(x))


def preprocess_pegasus(doccument):
    doccument = doccument.lower()
    doccument = " ".join(w for w in nltk.wordpunct_tokenize(doccument) \
         if w.lower() in words or not w.isalpha())
    doccument = replace_unneeded_items(doccument)
    return doccument


def remove_last_scentence_arvix(text):
    token_text = tokenize.sent_tokenize(text)
    if len(token_text) ==1:
        return text
    last_sentence = token_text[-1]
    if tokenize.word_tokenize(last_sentence)[-1] !='.':
        token_text = token_text[:-1]
        return " ".join([x for x in token_text])
    return text


def post_process_output_bigbird(output_text):
    output_text = output_text.replace("<s>", "").replace("</s>","").replace("<n>", "")
    output_text = output_text.replace("*","")
    output_text = output_text.split(". ")
    output_text = '. '.join(text.strip().capitalize() for text in output_text)
    output_text = output_text.replace(".aims", f".\nAims")
    output_text = output_text.replace(".results", f".\nResults")
    output_text = output_text.replace(".materials and methods", f".\nMaterials and methods")
    return output_text.rstrip()


if __name__ =="__main__":
    text_ip ="""In this Unit, we shall discuss about the broad spectrum of Indian political system, role and functions of political parties and conduct of elections at various levels. Also, we will look at some of the skills you need to have to be a good political reporter, especially how to develop contacts for newsgathering, and how to write in an informed way so that your readers or listeners can understand.
    You may find that the name of a local political leader has been published in the newspaper that you read. Read the news report to get the following details:
    The activity given above was a small exercise to make you understand what a political report is. But, a holistic understanding of political reporting is not possible without getting into the basics of parliamentary democracy, the Indian political system and various issues of governance. So, let us first look into them.
    Democracies follow two forms of political systems, Parliamentary Form (Britain) and Presidential Form (US). The Indian Constitution prescribes for parliamentary form of democracy in which the Parliament/Legislature assumes a very significant status in the political system. Why? Because the Executive (the Prime Minister and his Cabinet) is drawn from the Parliament/ Parliamentarians. The executive is accountable to the parliament. This is unlike the presidential form where the Executive (President and his Cabinet) is not drawn from the Parliament/ Parliamentarians, and is not accountable to the Parliament to that extent.
    Government (Ministers), Bills are drafted and brought to the parliament; and following parliamentary approval and presidential assent, the bills turn into Acts. The government governs with the help of these legislations and policy formulations.
    As stated above, the Parliament (and also the state legislatures) occupy an important position in the political system of India. People are always interested in knowing as to how the business is being transacted in the parliament and laws are being made on issues of peoples welfare. So, it becomes obligatory on the part of newspapers and news channels to report whatever is happening inside the Parliament during its working sessions.
    The Constitution of India provides for a bicameral Parliament consisting of the Lok Sabha (the House of the People) and Rajya Sabha (the Council of States). The Lok Sabha is composed of representatives of the people chosen through direct elections. The House, unless dissolved earlier, continues for five years from the date of its first meeting. The maximum strength of the Lok Sabha is 552.
    The Rajya Sabha consists of maximum 250 members. The seats have been allocated to various States and Union Territories, roughly in proportion to their population. The representatives of each state are elected by members of the respective Legislative Assemblies (Vidhan Sabhas) of the States in accordance with the system of proportional representation by means of a single transferable vote.
    There are dedicated reporters for covering the parliament as over the years parliamentary reporting has developed as an independent stream. Yet, every political reporter is supposed to follow the parliamentary proceedings. On many occasions, political news emanates from outside the Parliament even during the session. If political parties fail to vent their anger inside the House, they do it outside which is vital for political reporters.
    Political parties are the vehicles of democracy irrespective of whether a country has adopted the presidential system or parliamentary system. Understanding of issues related to political parties like their formation, election, registration and funding is necessary to every political reporter. Let us first define a political party.
    A political party is an organization or association of people who would have joined hands to contest elections and hold power in the government. Mostly political parties are formed with a purpose, each believing in a certain ideology. Every party agrees on some policies and programmes, with a view to promoting the collective good and/or taking care of the interests of their supporters.
    In democracies, political parties are elected by the electorate to run a government. India has multi-party system and as per latest Election Commission (EC) data (update till 15th March 2019), there are seven National political parties, 59 State parties and more than two thousand registered unrecognised parties. This number is abnormally high when compared with other democracies of the world. The issue of registration and recognition of political parties is being discussed in subsequent paragraphs.
    The Election Commission of India(ECI) has mandated regarding the constitution of political parties stipulating that they have to have clear rules regarding organizational elections at different levels and the periodicity of such elections and terms of office of the office-bearers of the party. However, if one looks at the situation on the ground, it is observed that most of parties do not hold regular election or follow erratic and arbitrary process in the conduct of elections. Thus, inner-party democracy is found to be weak.
    Beat Reporting-1 relevant papers including the party constitution. Without registration, no political party can contest an election. Registered parties get preference in the allotment of free symbols.
    Registered political parties can get recognition as State Party or National Party subject to fulfilment of conditions prescribed by the Election Commission. If a party is recognised as a State party, it is entitled for exclusive allotment of its reserved symbol to the candidates set up by it in the State/States in which it is so recognised; and if a party is recognised as a National party, it is entitled for exclusive allotment of its reserved symbol to the candidates set up by it throughout India.
    Recognised State and National parties need only one proposer for filing the nomination. They are also entitled for two sets of electoral rolls free of cost and broadcast/telecast facilities over the national broadcasters the All India Radio (AIR) and Doordarshan (DD) during general elections. Recognized parties are consulted by the Election Commission while finalising electoral rules and regulations and fixing schedules of polling.
    If a political party is treated as a recognised political party in four or more States, it is known as a National Party throughout India, but only so long as that political party continues to fulfil thereafter the conditions for recognition in four or more States on the results of any subsequent general election either to the House of the People or to the Legislative Assembly of any State. The status is reviewed periodically by the Election Commission.
    In India, political parties fund themselves through donations. Donors include party workers, businessmen and companies/corporate entities. Through amendment in Foreign Contributions (Regulation) Act 1976, the Government has allowed political parties to accept donations from foreign entities including companies with foreign shareholding.
    A few years back, parties were expected to file their Income Tax Returns (ITRs) containing details of donors who had made contributions above Rs 20,000. A copy of this return had to be sent to the Election Commission every year. However, the political parties used to declare most of their funds as having come from unnamed donors who donate less than Rs. 20,000 each, thereby getting away without naming the source of their donations and also enjoying tax exemptions.
    The Election Commission recommended that the law should be changed to make it mandatory for political parties to also disclose contributions less than Rs. 20,000. It favoured exemptions only upto Rs 2000. The Law Commision endorsed the recommendations of EC. As a result, the government amended the Income Tax Act and limited anonymous cash donations to Rs. 2000.
    Many political parties do not file ITRs, and even if they do, they never send the copy to the Commission. Alarge number of political parties do not participate in elections, and are alleged to be involved in turning black money to white. Under the existing laws, EC has the authority to register a political party but there is no provision to allow it to deregister any party that has been given recognition.
    You must be reading numerous political reports in the newspapers or news websites every day. Take the reference of a few news reports which appeared recently on the issues of funding and financial management of political parties. List the different aspects of financial management of political parties and write your opinion on them.
    Political scientists explain four functions of political parties: a) selection of candidates, b)mobilization of voters, c) facilitation of governance, d) monitoring the opposing party when it is in power. A political reporter is required not only to recognize these functions, but he/she has to appreciate the activities that the political parties undertake while performing these functions.
    As a political reporter, you may be covering mostly the activities of political parties. These activities, however, are not very well defined in a democratic set up. The basic objective of these activities is to attract the attention of media to get adequate media coverage and thereby reach out to the masses.
    The most common activities of political parties include street/corner meetings, public meetings, road shows and rallies. Public meetings are addressed by senior leaders whether belonging to the ruling party or parties in opposition. These are considered the most effective tool of connecting with the people who gather in large number to listen to their favourite leaders.
    Speeches made by political leaders in public meetings provide input for writing good news reports. So, a political reporter always keeps an eye on such activities. Public meeting is an all-weather activity though during elections it is a common sight. During elections, street/corners meetings are also held in large numbers.
    Rallies are organized by political parties as a show of their strength to their opponents. Rallies are attended by thousands of party workers as they spread in large areas for canvassing among the people to support their party. Bike rallies and cycle rallies are also quite popular. Road shows have become popular and is a comparatively a new form of political campaigning. Senior political leaders use this as a measure to reach out to the people and influence them.
    Beat Reporting-1 All these provide enough material to political reporters. In addition to all of the above, one peculiar Indian style of mass contact is Rath Yatra, where the political leader travels in a big vehicle along with his supporters and stops at different locations to address the people from an elevated pedestal fitted into the vehicle.
    Political parties in the opposition resort to demonstration, procession, dharna, road blockade and other similar activities for lodging their protest against the policies and actions of the government. During these activities, the protesters use plaques, banners, posters and black flags. They even raise slogans in support of their demands and to condemn their political opponents.
    Political parties make calls for strike or bandh (like Bharat Bandh), road blockade and other similar activities. On many occasions, these protests go ugly as the party workers turn berserk and indulge in violence. This invites police action which may involve lathi charge, shelling of tear gas or even firing. Sometimes, it may be the other way round. The police may obstruct the demonstration by using undue force, and in reaction the protesters turn violent and attack the police force. The protesters may even be seen pelting stones at the police.
    Contrary to this, silent processions are also taken out. Political workers are sometime found half-clad; they may even strip in protest. Effigy burning during demonstrations is also a common sight. Para-military forces or army may be deployed if the situation goes out of control and violence breaks out. For political reporters any such situation is big news; any injuries or causalities make the news even bigger.
    All major political parties hold press briefings at their national headquarters in New Delhi generally every day in the afternoon/evening. This is done to update the media about the stand the party would have taken on issues of topical importance or urgent nature. These briefings are also used to level charges on the political opponents and to defend the party if charges have been levelled against its own leaders.
    Press briefings are organized on a day-to-day basis at a fixed time, addressed by the official spokespersons of the party. On special occasions, senior leaders of the party also address the reporters and brief them. Political reporters make it a point to attend all such briefings.
    Press Conferences may also be organized as and when a party feels the need to share with the political reporters on some major issue. Unlike press briefings, the date, time and venue of the press conference has to be notified to the reporters.
    Every political party holds meetings at regular intervals including closed-door meetings and conferences or convention which are open to public view. The closed-door meetings include meetings of senior leaders of the party and meetings of the executive or working committee of the party. Political reporters find itdifficult to cover these meetings unless they are officially communicated by the party about the outcome of the meeting. Sometimes, a reporter may get some details of the meeting through a confidential source (may be a dissenting leader).
    On occasions, many political parties hold joint meetings for the formation of joint front or entering into an alliance. The opposition parties may also seek meetings with the President and the Prime Minister for submitting their representations on some issues.
    Most political parties also organize their annual conventions. Periodic conferences are also organized. Political reporters are given full or limited access to these conventions and conferences. This makes the coverage of these conventions and conferences convenient to the political reporter to access information.
    Political parties use social media platform as well for reaching out the people. Political reporters need to keep a track on such postings, as these may turn into important news stories. Political parties also publish their mouthpieces, like Kamal Sandesh of BJP and Congress Sandesh of the Congress. The reports and articles carried in these publications provide news material for political reporters.
    You would have noticed that most of the time in a year, some election is being held at one place or the other in the country. In fact, in any representative democracy elections are an integral part and hence, as a political reporter you will always be busy covering elections round the year. Also the Representation of the People Act 1951 has made specific provisions regarding elections. The Election Commission of India (ECI) has been entrusted with the task of conducting free and fair elections across the country.
    Universal Adult Franchise: The Constitution of India has prescribed for universal adult franchise, which means that every adult has a right to vote without any discrimination of caste, colour, religion or sex. The basic principle of one person one vote is followed. The voting right is based on the tenet of equality which is the cornerstone of democracy. As per the 61st ConstitutionalAmendment of 1988, the voting age has been reduced from 21 to 18 years which has really empowered the youth in the country. In 2019, the number of voters has increased to approximately 90 crore.
    Direct/Indirect Elections: The members of the legislature and the executive are elected through a well-established process of elections. The system is largely based on direct elections though some members of legislatures and executive functionaries are also elected through indirect elections. Direct election is where the electorates cast their votes for electing their representatives. In an indirect election, however, the elected representatives cast their votes for electing representatives of the State (RajyaSabha) and executive functionaries (President/ Vice President).
    Central/State/Local Elections: Broadly, elections take place at three levels; Central, State and Local (Local Self Government). At the Central level, elections for both the Houses of Parliament, the President and the Vice President are conducted; while elections for State Legislature takes place at the State level. At the Local level, elections are conducted for local self-government i.e. local bodies like Municipal Corporations and Panchayats. While elections at the Central and the State level are conducted by the Election Commission of India, elections of local bodies are conducted by the respective State Election Commissions.
    The Election Commission (EC) of India is an autonomous constitutional authority responsible for administering the election processes in India. It operates under the authority of the Constitution (Article 324) and the Representation of the People Act 1951.
    The Election Commission is empowered to conduct free and fair polls. It has powers under the Constitution to act in an appropriate manner, while the enacted laws make sufficient provisions to deal with a given situation in the conduct of an election.
    Structure of EC: Originally, the Election Commission had only one Commissioner, designated as the Chief Election Commissioner. Two additional commissioners were appointed to the Commission for the first time in October 1989, but they had a very short tenure, ending in January 1990. In 1993 the Commission become a multi-member body. The concept of a three-member Commission has been in operation since then, with the decisions being made by a majority vote.
    Powers of the EC: Though the Election Commission appears not having enough powers, but it acquires enormous powers during elections. The administrative machinery comes directly under the Election Commission as soon as the election schedule is notified in a state. The DM/Collector of a district assumes the role of District Election Officer during the election period. The police Chief of the district also report to the EC during this period. As per requirement of the Election Commission, the security forces are made available for election duties by the Central Government.
    Legal/Ethical Issues: The Election Commission has limited legal powers. It can reject nomination papers during scrutiny in case of inconsistency with rules. It can even disqualify a candidate for three years if details of election expenses are not submitted as stipulated by the Election Commission.
    Beat Reporting-1 The Election Commission issues a Model Code of Conduct for political parties and candidates during the election period. This code comes into effect with the announcement of election schedule by the Election Commission. However, there have been instances of violation of the code by various political parties with complaints being received for misuse of official machinery by the candidates.
    Indian Parliament has two houses, Lok Sabha (House of the People) and Rajya Sabha (Council of States). The term of Lok Sabha is five years, and General Election takes place before the expiry of this term. Every State has a Legislative Assembly called Vidhan Sabha (House of the People). Election to this House takes places exactly on the lines of Lok Sabha.
    Demarcation of constituencies: Under Article 82 of the Constitution, the Parliament enacts a DelimitationAct after every census. The election Commission demarcates parliamentary constituencies all across the country, and one member is declared elected from each constituency. The demarcation of a constituency depends upon the population of that geographical region, and the process is undertaken by constituting a Delimitation Commission.
    The present delimitation of constituencies has been done on the basis of 2001 Census figures under the provisions of Delimitation Act, 2002. Notwithstanding the above, the Constitution of India was specifically amended in 2002 not to have delimitation of constituencies till the first Census after 2026. Thus, the present Constituencies carved out on the basis of 2001 census shall continue to be in operation till the first census after 2026.
    Indirect Election: Indirect election takes place for Rajya Sabha. This upper house has a maximum strength of 250 members, of these 238 members are to be elected for a six-year term, with one-third retiring every two years. 12 members are to be nominated by the President of India. The members are indirectly elected, this being achieved by the votes of legislators in the State(s) and Union Territories.
    The elected members are chosen under the system of proportional representation by means of single transferable vote. The twelve nominated members are usually an eclectic mix of eminent artists (including actors), scientists, jurists, sportspersons, journalists and social workers.
    Some States in India have bicameral legislature. The other house is named as Vidhan Parishad (Legislative Council). Members to this house are also chosen through proportional representation system by means of single transferable vote.
    The President of India is also elected indirectly. Members of Parliament and members of State Legislatures cast their votes in this election under the system of proportional representation.
    Local Elections: State Election Commissions have been set up for conducting elections to local bodies as per Article 243K of the Constitution. This provision has been made through Constitutional (73rd Amendment) Act, passed in 1992, and implemented in 1993. Local bodies include Nagar Nigam, Nagar Mahapalika, Nagar Palika, ZilaParishad, District Panchayat, Gram Panchayat etc.
    The State Election Commissioner has several unique powers pertaining to the elections to local bodies. S/he chairs the Delimitation Commission which delimits local government constituencies and has full powers to conduct local government elections. S/he assigns reserved posts and constituencies and can disqualify candidates who do not submit election accounts.
    Our electoral process has been suffering from many problems right from the beginning. Hence, a process of electoral reforms was set in motion in 1990s. A law regarding the registration process for political parties was enacted in 1989. During the tenure of CEC T. N. Sheshan, reforms got a boost and photo I- Cards were issued to voters, system of revision of electoral lists was put in motion, and legal limits were fixed on the amount of money which a candidate can spend during election campaigns. This limit fixed differently for Lok Sabha and Vidhan Sabha, is periodically revised, and monitored by the Election Observers appointed by the Commission in every constituency.
    The Commission takes details of the candidates assets on an affidavit at the time of submitting the nomination paper. They are also required to give details of their expenditure within 30 days of the declaration of results. The campaign period has also been reduced by the Commission from 21 to 14 days for Lok Sabha and Assembly elections in order to cut down the election expenditure. Option of NOTA (none of the above) has also been introduced in 2013. The Commission can issue an order for prohibition of publication and dissemination of results of opinion polls and exit polls to prevent influencing the voting trends.
    However, the issue of paid news which is mushrooming and other issues such as funding of political parties, state funding of elections, compulsory voting, right to recall, simultaneous election to Lok Sabha and all Vidhan Sabhas are issues which are yet to be resolved.
    In news organizations  newspapers, radio channels and TV news channels  reporters are assigned beats for the smooth conduct of the reporting job. A beat is the subject area that a reporter is assigned to cover. The assignment of a particular beat helps the reporter to develop expertise in his/her area which includes expanding sources and gaining knowledge and experience for better performance.
    In newspapers, the reporting network comprises reporting broadly at three levels; City, State Capital and National Capital. Here, the city reporting or local reporting is referred to reporting from the city from where the newspaper edition is published. It is substituted by metro reporting in the case of reporting from metropolitan cities (if the newspaper is published from a metropolitan area). Every newspaper (from whichever city it is published) has a reporting set up in the state capital and the national capital. This reporting set up is called bureau.
    There are different beats distributed to reporters in the City Reporting Room, the State Bureau and the National Bureau. However, political beat is common at all the three levels, though the range of coverage differs. A political reporter in the local reporting room covers political activities happening in that particular city; while a reporter in the State Bureau covers political activities having wider reach including affairs related to the State Government. Areporter in the National Bureau covers national and international level politics including affairs related to the Central Government.
    Senior leaders and ministers: visits of the city by state and national level leaders of political parties, ministers of State/Central Government, Chief Minister, Prime Minister and President.
    In political newsgathering, reporters prefer to do field reporting or spot coverage. This includes reporting the sessions of legislatures, public meetings, speeches by political leaders, demonstrations and rallies, election campaigning and polling, counting and election results, conventions and meetings of the parties. On-the- spot interviews of political leaders or bites collected from them is considered an important assignment for political reporters.
    Thus, a political reporter has to do lot of running around. S/he has to follow a hectic schedule, and his/her job appears quite taxing. Mobility and connectivity are essential attributes of political reporters.
    Political reporters depend on official sources to a great extent. Press releases issued by political parties and statements issued by their leaders provide significant input for writing news reports. Political reporters never miss the press briefings of political parties and the press conferences organized by ministers and other political leaders. On many occasions, reporters are briefed about the outcome of some party meetings. In the wake of political controversies, political reporters wait for the official version issued by the concerned party.
    However, political reporters will not be able to have an edge unless they develop some confidential sources. These sources are mainly the insiders who give vital input to reporters. Such sources include dissenting leaders or political opponents. Political reporters need to cultivate confidential sources and should be prepared to protect their identity. Mutual trust is a must between a political reporter and his/her source.
    In political news writing, the same inverted pyramid news structure is adopted which you would have already gone through. Here, the most important information or fact comes first, followed by the less important information, continued by some extra information and followed by the least important facts. In writing lead of political news reports, the same 5 Ws and I H formula is applied.
    Writing Speeches/Press Conferences/Interviews: A political reporter is mostly found writing stories on speeches, press conferences or briefings and interviews. These news stories have a common element; they are one-source stories. Such stories are collection of direct and indirect quotations from the source.
    Speeches assume more significance if made by prominent political leaders and government functionaries. Areporter requires considerable skill to report a 5000- word speech in 300 words, and still give the reader an accurate report of what was said. The task becomes further difficult in the case of really lengthy speeches.
    All speeches, whether formal addresses on special occasions or impromptu remarks during an unstructured gathering, are handled very much alike. The reporter must consider the following three elements:
    A fourth consideration is the possible interpretation that any of the three elements may need. The proportion of the story to be devoted to each element varies with the comparative importance of each, but no story is complete without all three.
    The speaker should be properly identified in the lead. Sometime this can be done with a title or a short sentence. The reader needs to know who the speaker is and why his or her statements are worth quoting. Even a description of the speakers distinctive characteristics and manner of emphasizing certain points is sometimes woven into the story to give it more colour.
    The audience should also be described. How many people were there? Who were they? Why did they meet? The reporter needs to answer these questions in the news report. The names of persons in the audience need not be given unless it would be of interest to the reader to know the names of a few of the more prominent ones; but the reporter should tell whether they are bankers, teachers, doctors, engineers, farmers or party workers. Audience reaction is also worth noting.
    A speech is the most important of the three elements. What was the most important thing said by the speaker should be the first question that a reporter should ask himself while sitting down to write the story. However, two reporters, covering the same speech may not agree on what was the most important thing said. That is why the news reports on the same speech is published with different emphasis in different newspapers.
    In press conferences and interviews, only two elements, the speaker and the speech are considered important. Mostly, a political reporter writes the story in his/her own language and expression while taking care that the meaning of the statements made by the speaker does not change. Wherever required, the reporter decides to keep the statements in quotes. Though direct quotations tend to add emphasis, the reporter should avoid quoting routine, obvious or minor points from the speech or statements made by the speaker.
    Meetings and special events: News stories based on meetings, conferences, conventions and other political events form a major chunk of political reporting. There are so many meetings held by political parties every other day, but all of them are not worth reporting. There is no news value in the simple fact that a meeting or event has happened. A reporter should cover and write a story only when s/he is certain that the readers are interested in knowing about that meeting.
    Most political parties rely on newspapers for publicizing their meetings through a news report published in advance. The information required for writing such a news story includes the name of the party (if it is breakaway faction), the agenda of the meeting, names of leaders addressing or attending the meeting, time, date and place of the meeting. The more the reporter knows about the party and its leaders and the recent developments, the easier it will be for him to write an informative and interesting advance story.
    The job of a political reporter does not finish after a meeting would have taken place and it has been reported in newspapers. The reporter has to keep a track on the developments, if any, after the details/outcome/decisions of the meeting have been made public through news reports. The outcome of a meeting might be having far reaching consequences, and hence, a political reporter should always be vigilant on this count.
    Beat Reporting-1 2) In a democracy, opposition political parties resist the policies and actions of the government and/or the ruling party by raising their voice of protest. This can be done either inside the parliament and/or out on the street by organizing demonstrations and processions. Large participation in these activities is bound to mount pressure on the government/ruling party. Violence may mar many demonstrations/processions, though sometimes it can benefit the organizer in view of public sympathy and media coverage.
    """



    docs = helpers.split_doc(text_ip,4096)
    print(len(docs))
    result = ''
    for doc in docs:
        doc =preprocess_pegasus(doc)

        ############################################ bigbird-pegasus-large-arxiv ###############################################
        tokenizer = AutoTokenizer.from_pretrained("google/bigbird-pegasus-large-arxiv")

        model = BigBirdPegasusForConditionalGeneration.from_pretrained("google/bigbird-pegasus-large-arxiv")#attention_type="original_full"

        model = model.to("cuda")
        inputs = tokenizer(doc, return_tensors='pt', truncation=True, max_length=4096).to("cuda")
        prediction = model.generate(**inputs) #max_length output is 256
        prediction = tokenizer.batch_decode(prediction, no_repeat_ngram_size=3) #,

        prediction = post_process_output_bigbird(prediction[0])
        result += remove_last_scentence_arvix(post_process_output_bigbird(prediction)) + '\n'
    # return

    print(result)


############################################# bigbird-pegasus-large-pubmed ###############################################
# tokenizer = AutoTokenizer.from_pretrained("google/bigbird-pegasus-large-pubmed")

# model = BigBirdPegasusForConditionalGeneration.from_pretrained("google/bigbird-pegasus-large-pubmed", attention_type="original_full")

# model = model.to(device)

# inputs = tokenizer(text_ip, return_tensors='pt', max_length=4096, truncation=True).to(device)
# prediction = model.generate(**inputs)
# prediction = tokenizer.batch_decode(prediction,  no_repeat_ngram_size=3)
# prediction = post_process_output_bigbird(prediction[0])
#return 


######################################################################################################
# tokenizer = AutoTokenizer.from_pretrained("google/bigbird-pegasus-large-bigpatent")

# model = BigBirdPegasusForConditionalGeneration.from_pretrained("google/bigbird-pegasus-large-bigpatent", attention_type="original_full")

# model = model.to(device)
# inputs = tokenizer(text_ip, return_tensors='pt', max_length=4096, truncation=True).to(device)
# prediction = model.generate(**inputs)
# prediction = tokenizer.batch_decode(prediction,  no_repeat_ngram_size=3)
# prediction = post_process_output_bigbird(prediction[0])
# print(type(prediction))
# return
