
import matplotlib.pyplot as plt

def pie_plot(data, labels, title=None):
    plt.figure(figsize=(5,8))
    plt.pie(data, 
        labels=labels, 
       autopct = '%0.0f%%', shadow = 'True')
    if title:
        plt.title(title)
        
    plt.show()