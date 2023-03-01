#prints out a string
print('Hello World!\nMy name is HAL!\nWhat is your name?')

#Asks for user's name using input() function
def check_for_name(name):
    if name.strip().isdigit():
        print("Please enter your name not a number.")
        username = input('')
        check_for_name(username)
    else:
        print('Hello '+name+' nice to meet you!')
username = input('')
check_for_name(username)