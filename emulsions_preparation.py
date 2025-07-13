import numpy as np

### FUNCTIONS
def add_amount(total_solution, protein_perc, wanted_perc, oil_perc):
    protein = protein_perc*total_solution #g of protein in protein solution for protein_perc wt% protein
    buffer = total_solution - protein #g of buffer in protein solution for protein_perc wt% protein
    factor = np.ceil(protein_perc/wanted_perc) #to not use too much buffer
    print(r'The solution will be scaled down with a factor of {} to scale down the amount of buffer needed'.format(factor))

    protein = protein/factor #scale down to not overuse buffer
    buffer = buffer/factor #scale down to not overuse buffer

    total = protein+buffer
    fit = protein/wanted_perc-protein-buffer
    print(r'You need to add {}g of buffer to {}g of the protein/buffer solution for {}% for {}g total'.format(fit, total, wanted_perc*100, fit+total))
    
    total_solution = total+fit
    oil=(oil_perc*total_solution)/(1-oil_perc)
    print(r'Then {}g of oil needs to be added to get {}g total sample'.format(oil, oil+total_solution))
    return fit, oil

###VARIABLES
total_solution = 5 #desired grams of solution to make
protein_perc = 0.05 #of main solution
wanted_perc = 0.0025 #the desired percentage for the protein solution to dilute to
oil_perc = 0.05 #the desired oil percentage for the emulsion
 
###CALCULATE AMOUNTS TO ADD
add_buffer, add_oil = add_amount(total_solution, protein_perc, wanted_perc, oil_perc)



