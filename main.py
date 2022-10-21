import tkinter
from doctest import master
from tkinter.ttk import Combobox
import PIL
import keyboard
from ast import Or
from tkinter import *
import json
from tkinter import messagebox
import os
import time
from tkinter import filedialog
import tk as tk
from PIL import ImageTk, Image
import cv2
import numpy as np
import mediapipe as mp
# import dlib
import matplotlib.pyplot as plt
# from okk import *


top = Tk()
file_name=""
state =0


IMAGES_DIRECTORY = ''
OUTPUT_DIRECTORY = 'noise/'
# IMAGE_NAMES = [
#     image,
# ]

BALANCE_ALPHA = 0.2



showww=None
showww2=None
file_done=''


def edge_detection():
    # convert image to grayscale
    global file_name,file_done
    imagee = cv2.imread(file_name,0)
    imagee = cv2.cvtColor(imagee, cv2.COLOR_BGR2RGB)
    # blur image using a 5x5 kernel
    blur = cv2.GaussianBlur(imagee, (5, 5), 0)
    # apply canny edge detection
    canny = cv2.Canny(blur, 100, 200)
    file_done='canny_image.jpg'
    cv2.imwrite(file_done,canny)
    print(file_done)
    show_done(file_done)


# khử nhiễu
def Loc_TKTT_max(ksize):
    global file_name, file_done
    img = cv2.imread(file_name, 0)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    m, n = img.shape
    img_ket_qua_anh_loc_max = np.zeros([m, n])
    h=(ksize -1) // 2
    padded_img = np.pad(img, (h, h), mode='reflect')
    for i in range(m):
        for j in range(n):
            vung_anh_kich_thuoc_k = padded_img[i:i+ksize,j:j+ksize]
            gia_tri_Max = np.max(vung_anh_kich_thuoc_k)
            img_ket_qua_anh_loc_max[i, j] = gia_tri_Max
    file_done = 'loc_tktt_max.jpg'
    cv2.imwrite(file_done, img_ket_qua_anh_loc_max)
    print(file_done)
    show_done(file_done)

def Loc_Trung_binh_cat_Alpha(ksize, alpha):
    global file_name, file_done
    img = cv2.imread(file_name, 0)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    m,n = img.shape
    img_LQ_Trung_binh_cat_Alpha = np.zeros([m, n])
    h = (ksize - 1) // 2
    d = int(ksize*ksize*alpha)
    padded_img = np.pad(img,(h,h),mode='reflect')
    for i in range(m):
        for j in range(n):
            vung_anh_kich_thuoc_k = padded_img[i:i+ksize,j:j+ksize].flatten()
            vung_anh_kich_thuoc_k.sort()
            vung_anh_kich_thuoc_con_lai = vung_anh_kich_thuoc_k[d//2:-d//2]
            img_LQ_Trung_binh_cat_Alpha[i,j] = np.sum(vung_anh_kich_thuoc_con_lai) / (m*n-d)
    file_done = 'img_LQ_Trung_binh_cat_Alpha.jpg'
    cv2.imwrite(file_done, img_LQ_Trung_binh_cat_Alpha)
    print(file_done)
    show_done(file_done)

def Loc_TKTT_min( ksize):
    global file_name, file_done
    img = cv2.imread(file_name, 0)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    m, n = img.shape
    img_ket_qua_anh_loc_min = np.zeros([m, n])
    h=(ksize -1) // 2
    padded_img = np.pad(img, (h, h), mode='reflect')
    for i in range(m):
        for j in range(n):
            vung_anh_kich_thuoc_k = padded_img[i:i+ksize,j:j+ksize]
            gia_tri_Min = np.min(vung_anh_kich_thuoc_k)
            img_ket_qua_anh_loc_min[i, j] = gia_tri_Min
    file_done = 'Loc_TKTT_min.jpg'
    cv2.imwrite(file_done, img_ket_qua_anh_loc_min)
    print(file_done)
    show_done(file_done)

def Loc_TKTT_trung_vi( ksize):
    global file_name, file_done
    img = cv2.imread(file_name, 0)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    m, n = img.shape
    img_ket_qua_anh_loc_Trung_vi= np.zeros([m, n])
    h=(ksize -1) // 2
    padded_img = np.pad(img, (h, h), mode='reflect')
    for i in range(m):
        for j in range(n):
            vung_anh_kich_thuoc_k = padded_img[i:i+ksize,j:j+ksize]
            gia_tri_TV = np.median(vung_anh_kich_thuoc_k)
            img_ket_qua_anh_loc_Trung_vi[i, j] = gia_tri_TV
    file_done = 'img_ket_qua_anh_loc_Trung_vi.jpg'
    cv2.imwrite(file_done, img_ket_qua_anh_loc_Trung_vi)
    print(file_done)
    show_done(file_done)


def Loc_TKTT_Midpoint(ksize):
    global file_name, file_done
    img = cv2.imread(file_name, 0)
    m, n = img.shape
    img_ket_qua_anh_loc_Midpoint = np.zeros([m, n])
    h=(ksize -1) // 2
    padded_img = np.pad(img, (h, h), mode='reflect')
    for i in range(m):
        for j in range(n):
            vung_anh_kich_thuoc_k = padded_img[i:i+ksize,j:j+ksize]
            gia_tri_Max = np.max(vung_anh_kich_thuoc_k)
            gia_tri_Min = np.min(vung_anh_kich_thuoc_k)
            img_ket_qua_anh_loc_Midpoint[i, j] = (gia_tri_Max + gia_tri_Min)/2
    file_done = 'img_ket_qua_anh_loc_Midpoint.jpg'
    cv2.imwrite(file_done, img_ket_qua_anh_loc_Midpoint)
    print(file_done)
    show_done(file_done)




def Loc_Thich_Nghi_Cuc_Bo( ksize, phuong_sai_nhieu):
    global file_name, file_done
    img = cv2.imread(file_name, 0)
    m, n = img.shape
    img_ket_qua_anh_loc = np.zeros([m, n])
    h=(ksize -1) // 2
    padded_img = np.pad(img, (h, h), mode='reflect')
    for i in range(m):
        for j in range(n):
            vung_anh_kich_thuoc_k = padded_img[i:i+ksize,j:j+ksize]
            phuong_sai_cuc_bo = np.var(vung_anh_kich_thuoc_k)
            gia_tri_TB_cuc_bo = np.mean(vung_anh_kich_thuoc_k)
            if gia_tri_TB_cuc_bo  > phuong_sai_nhieu :
                img_ket_qua_anh_loc[i,j] = gia_tri_TB_cuc_bo
            else:
                img_ket_qua_anh_loc[i,j] = padded_img[i,j] - int((phuong_sai_nhieu / phuong_sai_cuc_bo) * (padded_img[i,j] - gia_tri_TB_cuc_bo))
    file_done = 'Loc_Thich_Nghi_Cuc_Bo.jpg'
    cv2.imwrite(file_done, img_ket_qua_anh_loc)
    print(file_done)
    show_done(file_done)
def Loc_Trung_binh_Contraharmonic(ksize,Q):
    global file_name, file_done
    img = cv2.imread(file_name, 0)
    m, n = img.shape
    img_ket_qua_anh_loc = np.zeros([m, n])

    h=(ksize -1) // 2
    padded_img = np.pad(img, (h, h), mode='reflect')
    padded_img_bac_Q_cong_1 = np.power(padded_img, Q+1)
    padded_img_bac_Q = np.power(padded_img, Q)

    for i in range(m):
        for j in range(n):
            vung_anh_kich_thuoc_k_bac_Q_cong_1 = padded_img_bac_Q_cong_1[i:i+ksize,j:j+ksize]
            vung_anh_kich_thuoc_k_bac_Q = padded_img_bac_Q[i:i + ksize, j:j + ksize]
            img_bac_Q_1 = np.sum(vung_anh_kich_thuoc_k_bac_Q_cong_1)
            img_bac_Q = np.sum(vung_anh_kich_thuoc_k_bac_Q)
            gia_tri_loc = img_bac_Q_1/img_bac_Q
            img_ket_qua_anh_loc[i, j] = gia_tri_loc
    file_done = 'Loc_Trung_binh_Contraharmonic.jpg'
    cv2.imwrite(file_done, img_ket_qua_anh_loc)
    print(file_done)
    show_done(file_done)
# def Loc_Trung_binh_Harmonic(img, ksize):
#     global file_name, file_done
#     img = cv2.imread(file_name, 0)
#     m, n = img.shape
#     img_ket_qua_anh_loc = np.zeros([m, n])
#     h=(ksize -1) // 2
#     padded_img = np.pad(img, (h, h), mode='reflect')
#     for i in range(m):
#         for j in range(n):
#             vung_anh_kich_thuoc_k = padded_img[i:i+ksize,j:j+ksize]
#             gia_tri_TB_cuc_bo = np.mean(vung_anh_kich_thuoc_k)
#             gia_tri_loc = np.sum(m*n/1/vung_anh_kich_thuoc_k)
#             if gia_tri_loc > gia_tri_TB_cuc_bo:
#                 img_ket_qua_anh_loc[i, j] = gia_tri_TB_cuc_bo
#             else:
#                 img_ket_qua_anh_loc[i, j] = gia_tri_loc
#     return img_ket_qua_anh_loc
# file_done = 'img_ket_qua_anh_loc_Midpoint.jpg'
#     cv2.imwrite(file_done, img_ket_qua_anh_loc_Midpoint)
#     print(file_done)
#     show_done(file_done)
def Loc_Trung_binh_hinh_hoc( ksize):
    global file_name, file_done
    img = cv2.imread(file_name, 0)
    m, n = img.shape
    img_ket_qua_anh_loc = np.zeros([m, n])
    h=(ksize -1) // 2
    padded_img = np.pad(img, (h, h), mode='reflect')
    for i in range(m):
        for j in range(n):
            vung_anh_kich_thuoc_k = padded_img[i:i+ksize,j:j+ksize]
            gia_tri_TB_cuc_bo = np.mean(vung_anh_kich_thuoc_k)
            gia_tri_loc = np.prod(vung_anh_kich_thuoc_k) ** (1.0 / m * n)
            if gia_tri_loc > gia_tri_TB_cuc_bo:
               img_ket_qua_anh_loc[i, j]= int(gia_tri_TB_cuc_bo)
            else:
               img_ket_qua_anh_loc[i,j] = int(gia_tri_loc)
    file_done = 'Loc_Trung_binh_hinh_hoc.jpg'
    cv2.imwrite(file_done, img_ket_qua_anh_loc)
    print(file_done)
    show_done(file_done)
def Loc_Trung_binh_so_hoc( ksize):
    global file_name, file_done
    img = cv2.imread(file_name, 0)
    m, n = img.shape
    img_ket_qua_anh_loc = np.zeros([m, n])
    h=(ksize -1) // 2
    padded_img = np.pad(img, (h, h), mode='reflect')
    for i in range(m):
        for j in range(n):
            vung_anh_kich_thuoc_k = padded_img[i:i+ksize,j:j+ksize]
            img_ket_qua_anh_loc[i,j] = np.mean(vung_anh_kich_thuoc_k)
    file_done = 'Loc_Trung_binh_so_hoc.jpg'
    cv2.imwrite(file_done, img_ket_qua_anh_loc)
    print(file_done)
    show_done(file_done)






def show_done(tmp):
    global showww2
    print(tmp)
    img_nhieu_tmp = cv2.imread(tmp,1)
    img_nhieu = cv2.resize(img_nhieu_tmp, dsize=None, fx=0.5, fy=0.5)
    img_nhieu = cv2.cvtColor(img_nhieu, cv2.COLOR_BGR2RGB)
    showww2 = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(img_nhieu))
    canvas2.create_image(0, 0, image=showww2, anchor=tkinter.NW)
def clickExit():
    top.quit()

def openfn():
    filename = filedialog.askopenfilename(title='open')
    return filename

def clickChoose():
    global file_name
    file_name = openfn()
    return file_name
def show_img():
    global file_name,showww
    clickChoose()
    img=cv2.imread(file_name,1)
    img_show=cv2.resize(img,dsize=None,fx=0.5,fy=0.5)
    img_show=cv2.cvtColor(img_show,cv2.COLOR_BGR2RGB)
    showww=PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(img_show))
    canvas1.create_image(0,0,image=showww, anchor = tkinter.NW)

def clickSave():
    global file_name
    if(combo.get()=='chọn'):
        messagebox.showinfo("Title", "Hãy chọn chức năng")
    else :
        if file_name == "":
            messagebox.showinfo("Title", "Hãy chọn file ")
        else:
            global top
            currentTime = time.time() * 1000
            new_name = file_name.split("/")
            Label(top, text="Hoàn thành lọc ảnh (ảnh được lưu tại /noiser/" + new_name[len(new_name) - 1] + "_" + str(currentTime) + ".jpg )").place(x=50, y=120)

            if (combo.get() == 'Loc_TKTT_max'):
                print('Loc_TKTT_max')
                Loc_TKTT_max(7)
            elif (combo.get() == 'edge detection'):
                print('bài 2')
                edge_detection()
                # show_done(canny_image)
            elif (combo.get() == 'Loc_Trung_binh_cat_Alpha'):
                Loc_Trung_binh_cat_Alpha(5,0.25)
                print('bài 3')
            elif (combo.get() == 'Loc_TKTT_min'):
                Loc_TKTT_min(7)
                print('bài 4')
            elif (combo.get() == 'Loc_TKTT_trung_vi'):
                Loc_TKTT_trung_vi(7)
                print('bài 5')
            elif (combo.get() == 'Loc_TKTT_Midpoint'):
                Loc_TKTT_Midpoint(5)
                print('bài 6')
            elif (combo.get() == 'Loc_Thich_Nghi_Cuc_Bo'):
                Loc_Thich_Nghi_Cuc_Bo(7,0.15)
                print('bài 7')
            elif (combo.get() == 'Loc_Trung_binh_Contraharmonic'):
                Loc_Trung_binh_Contraharmonic(7,0.15)
                print('bài 7')
            elif (combo.get() == 'Loc_Trung_binh_so_hoc'):
                Loc_Trung_binh_so_hoc(5)
                print('bài 7')
            elif (combo.get() == 'Loc_Trung_binh_hinh_hoc'):
                Loc_Trung_binh_hinh_hoc(3)
                print('bài 7')

#
# def predict_image(cascade, src, scale_percent=100):
#     # calculate the 50 percent of original dimensions
#     width = int(src.shape[1] * scale_percent / 100)
#     height = int(src.shape[0] * scale_percent / 100)
#     # dsize
#     dsize = (width, height)
#     resized = cv2.resize(src, dsize)
#
#     # gray_scale image
#     gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
#     try:
#         faces = cascade.detectMultiScale(gray, 1.1, 4)
#         for (x, y, w, h) in faces:
#             cv2.rectangle(resized, (x, y), (x + w, y + h), (98, 96, 2), 5)
#     except:
#         pass
#     return resized
# def load_model(filename):
#     cascade = cv2.CascadeClassifier(filename)
#     return cascade
# def detection():
#     face_cascade = load_model('models/haarcascade_frontalface_alt2.xml')
#     mg = cv2.imread('./test/img.jpg')
#     imageTest = predict_image(face_cascade, mg)



def infoo():
    img = Image.open("bg.png")
    img = ImageTk.PhotoImage(img)
    panel = Label(top, image=img)
    panel.image = img
    panel.place(x=0, y=0)

    btnExit = Button(top, text="   Exit  ", fg="red", font=(25), bg='pink', command=clickExit).place(x=1400, y=70)
    btnback = Button(top, text="   back   ", fg="red", font=(25), bg='pink', command=load_main).place(x=1300, y=70)
    lbl = Label(top, text="MÔN HỌC: XỬ LÝ ẢNH", font="-weight bold", bg='pink').place(x=700, y=150)
    lbl = Label(top, text="GIẢNG VIÊN: XXXXXXXXXXXXXXXXX", font=(65), bg='pink').place(x=850, y=200)
    lbl = Label(top, text="NHÓM XXXXXXXX",font="-weight bold", bg='pink').place(x=700, y=300)

    lbl = Label(top, text="Thẩm Ngọc Ánh                ", font=(65), bg='pink').place(x=400, y=350)
    lbl = Label(top, text="Nguyễn Văn Kiên              ", font=(65), bg='pink').place(x=400, y=400)
    lbl = Label(top, text="Nguyễn Trung Nguyên          ", font=(65), bg='pink').place(x=400, y=450)
    lbl = Label(top, text="Nguyễn Trọng Sơn             ", font=(65), bg='pink').place(x=400, y=500)
    lbl = Label(top, text="Văn Dương Thanh Toán         ", font=(65), bg='pink').place(x=400, y=550)
    lbl = Label(top, text="Nguyễn Thị Thanh Tuyền       ", font=(65), bg='pink').place(x=400, y=600)
    lbl = Label(top, text="Nguyễn Lam Trường            ", font=(65), bg='pink').place(x=400, y=650)
    lbl = Label(top, text="N19DCCN012", font=(65), bg='pink').place(x=1000, y=350)
    lbl = Label(top, text="N19DCCN077", font=(65), bg='pink').place(x=1000, y=400)
    lbl = Label(top, text="N19DCCN124", font=(65), bg='pink').place(x=1000, y=450)
    lbl = Label(top, text="N19DCCN161", font=(65), bg='pink').place(x=1000, y=500)
    lbl = Label(top, text="N19DCCN172", font=(65), bg='pink').place(x=1000, y=550)
    lbl = Label(top, text="N19DCCN183", font=(65), bg='pink').place(x=1000, y=600)
    lbl = Label(top, text="N19DCCN218", font=(65), bg='pink').place(x=1000, y=650)
def load_main():
    global lbl,canvas1,canvas2,combo

    img = Image.open("bg.png")
    img = ImageTk.PhotoImage(img)
    panel = Label(top, image=img)
    panel.image = img
    panel.place(x=0,y=0)

    # canvas1 = Canvas(top, width=1500, height=800, bg='pink')
    # canvas1.place(x=0, y=0)
    canvas1 = Canvas(top, width=600, height=600)
    canvas1.place(x=50, y=140)
    canvas2 = Canvas(top, width=600, height=600)
    canvas2.place(x=750, y=140)
    btnExit = Button(top, text="   Exit  ", fg="red", font=(45), bg='pink', command=clickExit).place(x=1400, y=70)
    btnChoose = Button(top, text="open", fg="red", font=(25), bg='pink', command=show_img).place(x=1200, y=70)
    btnSave = Button(top, text="   Run   ", fg="red", font=(25), bg='pink', command=clickSave).place(x=1300, y=70)
    btninfo = Button(top, text="   info   ", fg="red", font=(25), bg='pink', command=infoo).place(x=1100, y=70)
    combo = Combobox(top)
    combo['value'] = ('chọn',
                      'Loc_TKTT_max',
                      'Loc_TKTT_min',
                      'Loc_TKTT_trung_vi',
                      'Loc_TKTT_Midpoint',
                      'Loc_Trung_binh_cat_Alpha',
                      'Loc_Thich_Nghi_Cuc_Bo',
                      'Loc_Trung_binh_Contraharmonic',
                      'Loc_Trung_binh_so_hoc',
                      'Loc_Trung_binh_hinh_hoc',
                      'edge detection')
    combo.place(x=570, y=70)
    combo.current(0)
    Label(top, text="ảnh gốc", font="-weight bold").place(x=250, y=750)
    Label(top, text="ảnh đã xửa lý", font="-weight bold").place(x=1100, y=750)


if __name__ =="__main__":
    top.geometry("1500x800")
    top.title("Bài thực hành")
    top.configure(background="gray")
    # Add image file
    print("oke")
    load_main()
    top.mainloop()