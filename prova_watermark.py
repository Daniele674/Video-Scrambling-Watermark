from blind_watermark import WaterMark, blind_watermark

blind_watermark.bw_notes.close()
bwm1 = WaterMark(password_img=1, password_wm=1)
bwm1.read_img('testSet/Gatto_europeo4.jpg')
wm = 'viva il mondo'
bwm1.read_wm(wm, mode='str')
bwm1.embed('embedded.png')
len_wm = len(bwm1.wm_bit)
# print('Put down the length of wm_bit {len_wm}'.format(len_wm=len_wm))

bwm1 = WaterMark(password_img=1, password_wm=1)
wm_extract = bwm1.extract('embedded.png', wm_shape=len_wm, mode='str')
print(f'Il watermark estratto è: {wm_extract}')