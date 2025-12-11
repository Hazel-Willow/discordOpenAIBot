# discordOpenAIBot
An LLM bot for discord with support for LMStudio and Ollama for the backend
Uses standard OpenAI API calls with /v1 formatting

Will work on Ubuntu or Windows; however, the instructions are written for Ubuntu for now

# Installation:
## Dependencies:
First you're gonna need your standard python and dev tools; python3, pip, git, and venv

`sudo apt install build-essential libssl-dev libffi-dev python3-dev python3-pip python3-venv`

Next up, you're gonna wanna go out to grab Ollama or LM Studio. If you're running an NVidia card, I recommend LM Studio, otherwise Ollama plays nice with the Mesa drivers

Links:
https://ollama.com/download
https://lmstudio.ai/

Install instructions for those at their respective websites, and I recommend running some sanity checks after install to make sure your LLMs are working. LM Studio will require you to start the server for the python script to access it; Ollama will not

## Bot Installation:
Next up is creating a directory for your bot, in your home folder is fine

`mkdir ~/bots`

Clone the github project

`git clone https://github.com/Hazel-Willow/discordOpenAIBot.git ~/bots`

Traverse into the directory

`cd ~/bots`

Set up a venv

`python3 -m venv BotProject`

Enter your venv

`source BotProject/bin/activate`

Install the required python stuff

`pip install -r ./req*`

## Configuring the bot with the config.yaml file
I'm going to be filling this out in more detail later, including how to set things up in the discord dev portal; but, for now I'm going to let you google that part of it and just go over some of the config.yaml file

`bot_token` is your bot's Client Secret, this only gets shown to you once in the discord dev portal, so make sure you grab it then

`client_id` is the Client ID nest to the Client Secret

`nickname` is just that--a case sensitive nickname that will get your bot to respond without a direct ping

`system_prompt` is kinda the biography of your bot, it's gonna inform the bot of it's identity. Some tips on writing that, use declarative statements, and write it like you're talking at a person. "You are a stuffy accountant who is incredibly bad at math." Things like that

Good luck, have fun!
I'll be back later to update this readme.md

~Hazel Willow




