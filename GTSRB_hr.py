
# coding: utf-8

# # Attack Examples on GTSRB

# In[1]:


# Specify visible cuda device
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# In[2]:


# get_ipython().run_line_magic('matplotlib', 'inline')

# get_ipython().run_line_magic('reload_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')

from parameters import *
from lib.utils import *
from lib.attacks import *
from lib.keras_utils import *
from lib.RandomTransform import *
from lib.OptCarlini import *
from lib.OptTransform import *


# ### Initialize Model

# In[3]:


# Build and load trained model
model = build_mltscl()
model.load_weights(WEIGTHS_PATH)

# Load dataset
x_train, y_train, x_val, y_val, x_test, y_test = load_dataset_GTSRB(
    n_channel=N_CHANNEL)

# Convert labels to one-hot encoding
y_train = keras.utils.to_categorical(y_train, NUM_LABELS)
y_test = keras.utils.to_categorical(y_test, NUM_LABELS)
y_val = keras.utils.to_categorical(y_val, NUM_LABELS)

# Read sign names
signnames = read_csv("./input_data/signnames.csv").values[:, 1]


# In[4]:


model.summary()


# ### Load data

# In[5]:


SAMPLE_IMG_DIR = './images/Original_Traffic_Sign_samples'
SAMPLE_LABEL = './images/Original_Traffic_Sign_samples/labels.txt'


# In[6]:


# Load sample images, labels and masks
x_smp, x_smp_full, y_smp, masks, masks_full = load_samples(SAMPLE_IMG_DIR, SAMPLE_LABEL)


# In[31]:


# Set target class to attack
tg = 10
print ("Target class: " + signnames[tg])
# Set number of samples
size = 10

y_target = np.zeros((len(x_test))) + tg
y_target = keras.utils.to_categorical(y_target, NUM_LABELS)

# Filter samples (originally misclassified, originally classified as target)
x_fil, y_fil, del_id = filter_samples(model, x_smp, y_smp, y_target=y_target)
x_fil_full = np.delete(x_smp_full, del_id, axis=0)
masks_fil = np.delete(masks, del_id, axis=0)
masks_fil_full = np.delete(masks_full, del_id, axis=0)

# Set samples to attack (choose some samples by random)
ind = np.random.choice(range(len(y_fil)), size=size)
x_ben = np.copy(x_fil[ind])
x_ben_full = np.copy(x_fil_full[ind])
y_ben = np.copy(y_fil[ind])
y_tg = np.copy(y_target[ind])
masks_ben = np.copy(masks_fil[ind])
masks_ben_full = np.copy(masks_fil_full[ind])


# ## Attack Examples
# 
# ### Fast Gradient

# In[48]:


# Specify list of magnitudes
mag_list = np.linspace(1.0, 2.0, 6)
x_fg = fg(model, x_ben, y_tg, mag_list, target=True, mask=masks_ben)


# In[61]:


im = x_ben[1]
print ("Original class: " + signnames[predict(model, im)])
plt.imshow(im)
plt.axis('off')
plt.show()

im = x_fg[5, 1]
print ("Adversarial class: " + signnames[predict(model, im)])
plt.imshow(im)
plt.axis('off')
plt.show()


# ### Iterative Attack
# Iterative steps in gradient direction

# In[53]:
'''

x_it = iterative(model, x_ben, y_tg, n_step=32, step_size=0.05, target=True, mask=masks_ben)


# In[54]:


im = x_ben[0]
print "Original class: " + signnames[predict(model, im)]
plt.imshow(im)
plt.axis('off')
plt.show()

im = x_it[0]
print "Adversarial class: " + signnames[predict(model, im)]
plt.imshow(im)
plt.axis('off')
plt.show()


# ### Optimize Attack

# In[55]:


# Initialize optimizer
opt = OptCarlini(model, c=1, lr=0.01, target=True, use_bound=False, init_scl=0.1,
                 loss_op=0, var_change=True, k=5)
# Run optimizer on sample (only take one sample at a time)
x_adv, norm = opt.optimize(x_ben[0], y_tg[0], n_step=5000, prog=True, mask=masks_ben[0])
# Run optimier with constant search
#x_adv, norm = opt.optimize_search(x_ben[0], y_tg[0], n_step=5000, search_step=10, prog=True, mask=masks_ben[0])


# In[56]:


im = x_ben[0]
print "Original class: " + signnames[predict(model, im)]
plt.imshow(im)
plt.axis('off')
plt.show()

im = x_adv
print "Adversarial class: " + signnames[predict(model, im)]
plt.imshow(im)
plt.axis('off')
plt.show()


# ### Optimize with Transformation

# In[57]:


# Initialize optimizer
opt = OptTransform(model, c=1, lr=0.01, target=True, use_bound=False, init_scl=0.1,
                   loss_op=0, var_change=True, k=5, batch_size=32)
# Run optimizer on sample
x_adv, norm = opt.optimize(x_ben[0], y_tg[0], n_step=5000, prog=True, mask=masks_ben[0])
# Run optimier with constant search
#x_adv, norm = opt.optimize_search(x_ben[0], y_tg[0], n_step=5000, search_step=10, prog=True, mask=masks_ben[0])


# In[58]:


im = x_ben[0]
print "Original class: " + signnames[predict(model, im)]
plt.imshow(im)
plt.axis('off')
plt.show()

im = x_adv
print "Adversarial class: " + signnames[predict(model, im)]
plt.imshow(im)
plt.axis('off')
plt.show()


# In[60]:


# Evaluate each attack, return a list of adv success rate
print eval_adv(model, x_fg, y_tg, target=True)
print eval_adv(model, x_it, y_tg, target=True)


# ## Appendix
# 
# ### Model trainer

# In[7]:


# Build model
model = built_mltscl()

# Load dataset
x_train, y_train, x_val, y_val, x_test, y_test = load_dataset_GTSRB(
    n_channel=N_CHANNEL, train_file_name='train_extended.p')

y_train = keras.utils.to_categorical(y_train, NUM_LABELS)
y_test = keras.utils.to_categorical(y_test, NUM_LABELS)
y_val = keras.utils.to_categorical(y_val, NUM_LABELS)


# In[10]:


filepath = './weights.{epoch:02d}-{val_loss:.2f}.hdf5'
modelCheckpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, 
                                                  save_best_only=False, save_weights_only=False, 
                                                  mode='auto', period=1)
earlyStop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, 
                                          verbose=0, mode='auto')


# In[14]:


model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCH, verbose=1, 
          callbacks=[modelCheckpoint, earlyStop], validation_data=(x_val, y_val), 
          shuffle=True, initial_epoch=0)
'''
