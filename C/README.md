> To run the code we need some softwares and tehnologies:
- Install PyCharm IDE. You can find it on Jetbrains link:
https://www.jetbrains.com/fr-fr/pycharm/download/#section=windows
- Install git. You can find an easy tutorial to install git with the link:
https://git-scm.com/downloads
- Install Docker, You can follow the tutorial on docker website it is really well explained
https://www.docker.com/products/docker-desktop

>If you already have the full project on your desktop, skip this step: \
> Open a command prompt where you want the project to be cloned and clone the project using the command below:
- git clone https://github.com/CANEVETGASPARD/HW2_CANEVET.git

> Then load the input files with the link given on isis and put them in a folder named "recommendation-datasets".

> If you have already created a python interpreter using the DockerFile from the project, skip the next three steps.

> Then open HW2_CANEVET project with PyCharm and use terminal located on the bottom of the interface.
Your terminal path should end with "HW2_CANEVET>", otherwise use the cd command to move to this location.
 
> Then run the command: 
- docker build -t hw2 . 
  
It will build a docker image named hw2 with Dockerfile from our project. 

> Then create a Python interpreter linked to your Docker. Find a simple tutorial on the link:
- https://www.jetbrains.com/help/pycharm/using-docker-as-a-remote-interpreter.html

You have to adapt it to our project. Thus, when you choose the docker image for the interpreter, you have to choose hw2 docker image.

> Then create a python configuration to run "the data_exploration.py" and "recommendation_system.py" python code using the specific interpreter. \
On the top right of the interface, click on edit configuration > add configuration > name your configuration, choose the algorithm path you want to use, use the docker interpreter and set the working directory to the HW2_CANEVET folder path.

Our project is now set up we just have to run the configuration.