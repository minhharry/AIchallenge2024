res = []
for segment in 'Hello. World.'.split('.'):
    if len(segment) > 2:
        res.append(segment)
print(res)