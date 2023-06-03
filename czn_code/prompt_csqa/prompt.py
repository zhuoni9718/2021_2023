def getprompt(promptname):
    # 如果promptname需要检索? 或者需要组合 我写过检索吧
    PROMPT = {'zeroshot':"Generate some knowledge about the sentence in the input. Examples:\n\
Input: {}\n\
Knowledge:",
'cg':"generate a sentence with:{}",
 'Q_wrong_K':"Generate some knowledge about the sentence in the input. Examples:\n\
Input: Google Maps and other highway and street GPS services have replaced what?\n\
Knowledge: Wings is used to fly.\n\
Input: The fox walked from the city into the forest, what was it looking for?\n\
Knowledge: The job of body guards is to ensure the safety and security of the employer.\n\
Input: You can share files with someone if you have a connection to a what?\n\
Knowledge: Speeches can share knowledge and insights with others.\n\
Input: Too many people want exotic snakes. The demand is driving what to carry them?\n\
Knowledge: There are red flowers in the garden where bees gather nectar.\n\
Input: The body guard was good at his duties, he made the person who hired him what?\n\
Knowledge: Skiing is good for keeping body and mind healthy.\n\
Input: {}\n\
Knowledge:",
        'QK':"Generate some knowledge about the sentence in the input. Examples:\n\
Input: Google Maps and other highway and street GPS services have replaced what?\n\
Knowledge: Electronic maps are the modern version of paper atlas.\n\
Input: The fox walked from the city into the forest, what was it looking for?\n\
Knowledge: Natural habitats are usually away from cities.\n\
Input: You can share files with someone if you have a connection to a what?\n\
Knowledge: Files can be shared over the Internet.\n\
Input: Too many people want exotic snakes. The demand is driving what to carry them?\n\
Knowledge: Some people raise snakes as pets.\n\
Input: The body guard was good at his duties, he made the person who hired him what?\n\
Knowledge: The job of body guards is to ensure the safety and security of the employer.\n\
Input: {}\n\
Knowledge:",
'Q1K':"Generate some knowledge about the sentence in the input. Examples:\n\
Input: Google Maps and other highway and street GPS services have replaced what?\n\
Knowledge: Electronic maps are the modern version of paper atlas.\n\
Input: {}\n\
Knowledge:",
'Q2K':"Generate some knowledge about the sentence in the input. Examples:\n\
Input: Google Maps and other highway and street GPS services have replaced what?\n\
Knowledge: Electronic maps are the modern version of paper atlas.\n\
Input: The fox walked from the city into the forest, what was it looking for?\n\
Knowledge: Natural habitats are usually away from cities.\n\
Input: {}\n\
Knowledge:",
'Q3K':"Generate some knowledge about the sentence in the input. Examples:\n\
Input: Google Maps and other highway and street GPS services have replaced what?\n\
Knowledge: Electronic maps are the modern version of paper atlas.\n\
Input: The fox walked from the city into the forest, what was it looking for?\n\
Knowledge: Natural habitats are usually away from cities.\n\
Input: You can share files with someone if you have a connection to a what?\n\
Knowledge: Files can be shared over the Internet.\n\
Input: {}\n\
Knowledge:",
'Q4K':"Generate some knowledge about the sentence in the input. Examples:\n\
Input: Google Maps and other highway and street GPS services have replaced what?\n\
Knowledge: Electronic maps are the modern version of paper atlas.\n\
Input: The fox walked from the city into the forest, what was it looking for?\n\
Knowledge: Natural habitats are usually away from cities.\n\
Input: You can share files with someone if you have a connection to a what?\n\
Knowledge: Files can be shared over the Internet.\n\
Input: Too many people want exotic snakes. The demand is driving what to carry them?\n\
Knowledge: Some people raise snakes as pets.\n\
Input: {}\n\
Knowledge:",
'Q6K':"Generate some knowledge about the sentence in the input. Examples:\n\
Input: Google Maps and other highway and street GPS services have replaced what?\n\
Knowledge: Electronic maps are the modern version of paper atlas.\n\
Input: The fox walked from the city into the forest, what was it looking for?\n\
Knowledge: Natural habitats are usually away from cities.\n\
Input: You can share files with someone if you have a connection to a what?\n\
Knowledge: Files can be shared over the Internet.\n\
Input: Too many people want exotic snakes. The demand is driving what to carry them?\n\
Knowledge: Some people raise snakes as pets.\n\
Input: The body guard was good at his duties, he made the person who hired him what?\n\
Knowledge: The job of body guards is to ensure the safety and security of the employer.\n\
Input: The bird can fly with what?\n\
Knowledge: Wings is used to fly.\n\
Input: {}\n\
Knowledge:",
'Q7K':"Generate some knowledge about the sentence in the input. Examples:\n\
Input: Google Maps and other highway and street GPS services have replaced what?\n\
Knowledge: Electronic maps are the modern version of paper atlas.\n\
Input: The fox walked from the city into the forest, what was it looking for?\n\
Knowledge: Natural habitats are usually away from cities.\n\
Input: You can share files with someone if you have a connection to a what?\n\
Knowledge: Files can be shared over the Internet.\n\
Input: Too many people want exotic snakes. The demand is driving what to carry them?\n\
Knowledge: Some people raise snakes as pets.\n\
Input: The body guard was good at his duties, he made the person who hired him what?\n\
Knowledge: The job of body guards is to ensure the safety and security of the employer.\n\
Input: The bird can fly with what?\n\
Knowledge: Wings is used to fly.\n\
Input: I have something in my head I want to share, what ways can I do that?\n\
Knowledge: Speeches can share knowledge and insights with others.\n\
Input: {}\n\
Knowledge:",
'Q8K':"Generate some knowledge about the sentence in the input. Examples:\n\
Input: Google Maps and other highway and street GPS services have replaced what?\n\
Knowledge: Electronic maps are the modern version of paper atlas.\n\
Input: The fox walked from the city into the forest, what was it looking for?\n\
Knowledge: Natural habitats are usually away from cities.\n\
Input: You can share files with someone if you have a connection to a what?\n\
Knowledge: Files can be shared over the Internet.\n\
Input: Too many people want exotic snakes. The demand is driving what to carry them?\n\
Knowledge: Some people raise snakes as pets.\n\
Input: The body guard was good at his duties, he made the person who hired him what?\n\
Knowledge: The job of body guards is to ensure the safety and security of the employer.\n\
Input: The bird can fly with what?\n\
Knowledge: Wings is used to fly.\n\
Input: I have something in my head I want to share, what ways can I do that?\n\
Knowledge: Speeches can share knowledge and insights with others.\n\
Input: Where do bees congregate with red flowers?\n\
Input: {}\n\
Knowledge:",
'Q9K':"Generate some knowledge about the sentence in the input. Examples:\n\
Input: Google Maps and other highway and street GPS services have replaced what?\n\
Knowledge: Electronic maps are the modern version of paper atlas.\n\
Input: The fox walked from the city into the forest, what was it looking for?\n\
Knowledge: Natural habitats are usually away from cities.\n\
Input: You can share files with someone if you have a connection to a what?\n\
Knowledge: Files can be shared over the Internet.\n\
Input: Too many people want exotic snakes. The demand is driving what to carry them?\n\
Knowledge: Some people raise snakes as pets.\n\
Input: The body guard was good at his duties, he made the person who hired him what?\n\
Knowledge: The job of body guards is to ensure the safety and security of the employer.\n\
Input: The bird can fly with what?\n\
Knowledge: Wings is used to fly.\n\
Input: I have something in my head I want to share, what ways can I do that?\n\
Knowledge: Speeches can share knowledge and insights with others.\n\
Input: Where do bees congregate with red flowers?\n\
Knowledge: There are red flowers in the garden where bees gather nectar.\n\
Input: What might be the result of a season of successful skiing?\n\
Knowledge: Skiing is good for keeping body and mind healthy.\n\
Input: {}\n\
Knowledge:",
'Q10K':"Generate some knowledge about the sentence in the input. Examples:\n\
Input: Google Maps and other highway and street GPS services have replaced what?\n\
Knowledge: Electronic maps are the modern version of paper atlas.\n\
Input: The fox walked from the city into the forest, what was it looking for?\n\
Knowledge: Natural habitats are usually away from cities.\n\
Input: You can share files with someone if you have a connection to a what?\n\
Knowledge: Files can be shared over the Internet.\n\
Input: Too many people want exotic snakes. The demand is driving what to carry them?\n\
Knowledge: Some people raise snakes as pets.\n\
Input: The body guard was good at his duties, he made the person who hired him what?\n\
Knowledge: The job of body guards is to ensure the safety and security of the employer.\n\
Input: The bird can fly with what?\n\
Knowledge: Wings is used to fly.\n\
Input: I have something in my head I want to share, what ways can I do that?\n\
Knowledge: Speeches can share knowledge and insights with others.\n\
Input: Where do bees congregate with red flowers?\n\
Knowledge: There are red flowers in the garden where bees gather nectar.\n\
Input: What might be the result of a season of successful skiing?\n\
Knowledge: Skiing is good for keeping body and mind healthy.\n\
Input: The person is laying on the beach, why would he do that?\n\
Knowledge: People like to lie on the beach in the sun.\n\
Input: {}\n\
Knowledge:",
    'SK':"Generate some knowledge about the sentence in the input. Examples:\n\
Input: Google Maps and other highway and street GPS services have replaced atlas.\n\
Knowledge: Electronic maps are the modern version of paper atlas.\n\
Input: The fox walked from the city into the forest, natural habitat was it looking for.\n\
Knowledge: Natural habitats are usually away from cities.\n\
Input: You can share files with someone if you have a connection to a computer network.\n\
Knowledge: Files can be shared over the Internet.\n\
Input:Too many people want exotic snakes. The demand is driving pet shops to carry them.\n\
Knowledge: Some people raise snakes as pets.\n\
Input: The body guard was good at his duties, he made the person who hired him feel safe.\n\
Knowledge: The job of body guards is to ensure the safety and security of the employer.\n\
Input: {}\n\
Knowledge:",
    "what":"Generate some knowledge about the sentence in the input. Examples:\n\
Input: What do people use to absorb extra ink from a fountain pen?  \n\
Knowledge: Blotting paper can absorb ink from a pen.\n\
Input: What type of person typically contracts illness?\n\
Knowledge: The elderly have low immunity and are prone to illness.\n\
Input: I have something in my head I want to share, what ways can I do that?\n\
Knowledge: Speeches can share knowledge and insights with others.\n\
Input:The sensor would just the distance then set off an alarm, the installation expert explained it was called a what kind of sensor?\n\
Knowledge:The range sensor determines the distance and alarms.\n\
Input: People can program with what?\n\
Knowledge: People can program with a computer.\n\
Input:{}\n\
Knowledge:",
    "why":"Generate some knowledge about the sentence in the input. Examples:\n\
Input: The man was eating lunch, but rushed when he looked at his watch, why did he rush?\n\
Knowledge: Being late for work can cause anxiety.\n\
Input: Why do people engage in chatting with friends in class?\n\
Knowledge: People chat with friends in class for socialization, distraction, and comfort\n\
Input: The person is laying on the beach, why would he do that?\n\
Knowledge: People like to lie on the beach in the sun.\n\
Input: A lonely man committed suicide, why would he do that?\n\
Knowledge: People try to kill themselves when they are extremely sad.\n\
Input: Why would professionals playing sports not be able to compete?\n\
Knowledge: Professionals who play sports are usually athletes who get injured.\n\
Input:{}\n\
Knowledge:",
    "where":"Generate some knowledge about the sentence in the input. Examples:\n\
Input: Sammy wanted to go to where the people were.  Where might he go?\n\
Knowledge: Populated area means the number of people per unit area is high.\n\
Input: To locate a choker not located in a jewelry box or boutique where would you go?\n\
Knowledge: Jewelry stores sell all kinds of jewelry and accessories.\n\
Input: Where can meat last a long time?\n\
Knowledge: The cold air in the refrigerator will keep the meat fresh.\n\
Input: Where do bees congregate with red flowers?\n\
Knowledge: There are red flowers in the garden where bees gather nectar.\n\
Input: The student needed to get some new pencils, where did he go?\n\
Knowledge: The store sells new stationery, including pencils, notebooks and so on.\n\
Input:{}\n\
Knowledge:",
    'QKK':"Generate some knowledge about the sentence in the input. Examples:\n\
Input: Google Maps and other highway and street GPS services have replaced atlas.\n\
Knowledge: Google Maps and other highway and street GPS services belong to electronic maps.\n\
Knowledge: Electronic maps are the modern version of paper atlas.\n\
Input: The fox walked from the city into the forest, natural habitat was it looking for.\n\
Knowledge: Foxes need natural habitats to survive.\n\
Knowledge: Natural habitats are usually away from cities.\n\
Input: You can share files with someone if you have a connection to a computer network.\n\
Knowledge: Files can be shared over the Internet.\n\
Knowledge: You must connect the Internet to use it first.\n\
Input:Too many people want exotic snakes. The demand is driving pet shops to carry them.\n\
Knowledge: Some people raise snakes as pets.\n\
Knowledge: Pet shops sell pets to people.\n\
Input: The body guard was good at his duties, he made the person who hired him feel safe.\n\
Knowledge: The job of body guards is to ensure the safety and security of the employer.\n\
Knowledge: People who are protected by body guards will feel safe.\n\
Input: {}\n\
Knowledge:",
    'QK_logic':"Generate some knowledge about the sentence in the input. Examples:\n\
Input: The bird can fly with what?\n\
Knowledge: Wings is used to fly.\n\
Input: Dogs dissipate heat through what?\n\
Knowledge: Dogs dissipate heat through their tongues.\n\
Input: Spiders can hunt with what?\n\
Knowledge: Spiders use spider web to hunt.\n\
Input: A turtle protect itself with what?\n\
Knowledge: A turtle uses its shell to protect itself?\n\
Input: People can program with what?\n\
Knowledge: People can program with a computer.\n\
Input: You can share files with someone if you have a connection to a what?\n\
Knowledge: Files can be shared over the Internet.\n\
Input:{}\n\
Knowledge:",
    "QK_dif_log":"Input: Fish can breathe in the water with what?\n\
Knowledge: Fish can breathe in the water with gill.\n\
Input: Fish have what dogs not have? \n\
Knowledge: Fish have gill dogs not have.\n\
Input: Bird can fly with wings, tiger can not fly for what?\n\
Knowledge: Tiger have not wings.\n\
Input:{}\n\
Knowledge:",
    "logic_abs":"Generate some knowledge about the sentence in the input. Examples:\n\
Input: What do people use to absorb extra ink from a fountain pen?  \n\
Knowledge: Blotting paper can absorb ink from a pen.\n\
Input: What type of person typically contracts illness?\n\
Knowledge: The elderly have low immunity and are prone to illness.\n\
Input: I have something in my head I want to share, what ways can I do that?\n\
Knowledge: Speeches can share knowledge and insights with others.\n\
Input:The sensor would just the distance then set off an alarm, the installation expert explained it was called a what kind of sensor?\n\
Knowledge:The range sensor determines the distance and alarms.\n\
Input:How does getting paid feel?\n\
Knowledge:Getting paid for work can make people happy and satisfied.\n\
Input:{}\n\
Knowledge:",
    "logic_con":"Generate some knowledge about the sentence in the input. Examples:\n\
Input: Google Maps and other highway and street GPS services have replaced what?\n\
Knowledge: Electronic maps have replaced traditional paper maps.\n\
Input: What home entertainment equipment requires cable?\n\
Knowledge: Cable television needs cables to work properly.\n\
Input: Where can meat last a long time?\n\
Knowledge: The cold air in the refrigerator will keep the meat fresh.\n\
Input: What group of musicians will include someone playing the cello?\n\
Knowledge: Symphonies usually contain the following instruments: cello, trombone, clarinet, etc.\n\
Input: Where do bees congregate with red flowers?\n\
Knowledge: There are red flowers in the garden where bees gather nectar.\n\
Input:{}\n\
Knowledge:",
    "logic_ana":"Generate some knowledge about the sentence in the input. Examples:\n\
Input: Where in Southern Europe would you find many canals?\n\
Knowledge: Much of Venice is separated by waterways.\n\
Input: The president is the leader of what institution?\n\
Knowledge: The president is the head of the system of government.\n\
Input: What might be the result of a season of successful skiing?\n\
Knowledge: Skiing is good for keeping body and mind healthy.\n\
Input: What would you do if you want to be able to earn money?\n\
Knowledge: Hard work makes money.\n\
Input: What is required to be good at playing tennis?\n\
Knowledge: It takes great skill to play a good game of tennis.\n\
Input:{}\n\
Knowledge:",
    "logic_ind":"Generate some knowledge about the sentence in the input. Examples:\n\
Input: Sammy wanted to go to where the people were.  Where might he go?\n\
Knowledge: Populated area means the number of people per unit area is high.\n\
Input: To locate a choker not located in a jewelry box or boutique where would you go?\n\
Knowledge: Jewelry stores sell all kinds of jewelry and accessories.\n\
Input: Of all the rooms in a house it was his favorite, the aromas always drew him to the what?\n\
Knowledge: The aromas of food came from the kitchen.\n\
Input: You can share files with someone if you have a connection to a what?\n\
Knowledge: Computer networks can be used to share files.\n\
Input: The student needed to get some new pencils, where did he go?\n\
Knowledge: The store sells new stationery, including pencils, notebooks and so on.\n\
Input:{}\n\
Knowledge:",
    "logic_cau":"Generate some knowledge about the sentence in the input. Examples:\n\
Input: The man was eating lunch, but rushed when he looked at his watch, why did he rush?\n\
Knowledge: Being late for work can cause anxiety.\n\
Input: What is eating too much dinner likely to result in?\n\
Knowledge: Overeating can cause stomach ache.\n\
Input: The person is laying on the beach, why would he do that?\n\
Knowledge: People like to lie on the beach in the sun.\n\
Input: A lonely man committed suicide, why would he do that?\n\
Knowledge: People try to kill themselves when they are extremely sad.\n\
Input: Why would professionals playing sports not be able to compete?\n\
Knowledge: Professionals who play sports are usually athletes who get injured.\n\
Input:{}\n\
Knowledge:",
    "logic_comprehensive_pre":"Generate some knowledge about the sentence in the input. Examples:\n\
Input: The person is laying on the beach, why would he do that?\n\
Knowledge: People like to lie on the beach in the sun.\n\
Input: What do people use to absorb extra ink from a fountain pen? \n\
Knowledge: Blotting paper can absorb ink from a pen.\n\
Input: What home entertainment equipment requires cable?\n\
Knowledge: Cable television needs cables to work properly.\n\
Input:{}\n\
Knowledge:",
    "logic_comprehensive":"Generate some knowledge about the sentence in the input. Examples:\n\
Input:The sensor would just the distance then set off an alarm, the installation expert explained it was called a what kind of sensor?\n\
Knowledge:The range sensor determines the distance and alarms.\n\
Knowledge: Symphonies usually contain the following instruments: cello, trombone, clarinet, etc.\n\
Input: Where do bees congregate with red flowers?\n\
Knowledge: The president is the head of the system of government.\n\
Input: What might be the result of a season of successful skiing?\n\
Input: To locate a choker not located in a jewelry box or boutique where would you go?\n\
Knowledge: Jewelry stores sell all kinds of jewelry and accessories.\n\
Input: The man was eating lunch, but rushed when he looked at his watch, why did he rush?\n\
Knowledge: Being late for work can cause anxiety.\n\
Input:{}\n\
Knowledge:",
    'commongen':"Generate some sentence about the concepts. Examples:\n\
Concepts: mountain ski skier\n\
Sentence: Skier skis down the mountain.\n\
Concepts: dog tail wag\n\
Sentence: he dog is wagging his tail.\n\
Concepts: canoe lake paddle\n\
Sentence: woman paddling canoe on a lake.\n\
Concepts: match stadium watch\n\
Sentence: soccer fans watches a league match in a stadium.\n\
Concepts: cat lick paw\n\
Sentence: A cat licks his paws.\n\
Concepts: {}\n\
Sentence:",
    'commongen2':"mountain ski skier = Skier skis down the mountain.\n\
dog tail wag = he dog is wagging his tail.\n\
canoe lake paddle = woman paddling canoe on a lake.\n\
match stadium watch = soccer fans watches a league match in a stadium.\n\
cat lick paw = A cat licks his paws.\n\
{} ="
}
    return PROMPT[promptname]