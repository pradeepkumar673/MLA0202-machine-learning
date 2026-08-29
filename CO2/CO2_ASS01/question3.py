data = [(1,1,'Yes'), (1,0,'Yes'), (0,1,'No'), (1,1,'Yes'), (0,0,'No')]
yes = [d for d in data if d[2]=='Yes']
no  = [d for d in data if d[2]=='No']
p_yes= len(yes)/len(data)
p_no =  len(no)/len(data)
p_fever1_yes = sum(1 for d in yes if d[0]==1)/len(yes)
p_head0_yes  = sum(1 for d in yes if d[1]==0)/len(yes)
p_fever1_no  = sum(1 for d in no if d[0]==1)/len(no)
p_head0_no   = sum(1 for d in no if d[1]==0)/len(no)
num_yes = p_yes*p_fever1_yes*p_head0_yes
num_no  = p_no*p_fever1_no*p_head0_no
total = num_yes+num_no

print('\n'*1)
print("1) Prior Probabilities: P(Yes) =", p_yes, " P(No) =", p_no)
print('\n')
print("2) Conditional Probabilities:")
print("P(Fever=1|Yes) =", p_fever1_yes, " P(Headache=0|Yes) =", p_head0_yes)
print(" P(Fever=1|No)  =", p_fever1_no, " P(Headache=0|No)  =", p_head0_no)
print('\n')
print("(3) P(Disease=Yes | Fever=1, Headache=0) =", num_yes/total)
print('\n')