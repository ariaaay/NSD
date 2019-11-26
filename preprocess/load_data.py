import nibabel as nib
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


img = nib.load("../data/betas_session01.nii.gz")
data = img.get_data()
print(data.shape)
plt.plot(data[50,50,50,:])
plt.show()