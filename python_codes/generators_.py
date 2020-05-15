def capitalize(values):
    # print(values)
    for value in values:
        # print(value)
        yield value.upper()

def hypenate(values):
    # print(values)
    for value in values:
        yield f"-{value}-"

def leetspeak(values):
    for value in values:
        if value in {'t', 'T'}:
            yield '7'
        elif value in {'e', 'E'}:
            yield '3'
        else:
            yield value

def join(values):
    print(values)
    return "".join(values)


print(join(capitalize("this will be uppercase text")))
print(join(leetspeak("This isn't a leetspeak")))
print(join(hypenate("will be hypenated by letters")))
print(join(hypenate("will be hypenated by words".split())))

# next() and send() 

def psychologist():
    print("Please tell me your problems")
    while True:
        answer = (yield)
        print("answer->", answer)
        if answer is not None:
            if answer.endswith('?'):
                print("Dont ask yourself too much questions")
            elif 'good' in answer:
                print("That's good, go on")
            elif 'bad' in answer:
                print('Dont be so negative')
            
free = psychologist()
print('free->', free)
print('next->', next(free))
print("send->", free.send("I feel bad"))
print("send->", free.send("Why i shouldnt ?"))
print("send ->", free.send("ok then i should find what is good for me"))
