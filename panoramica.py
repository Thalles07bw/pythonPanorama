import cv2
import os
import numpy as np

pastaPrincipal = 'Test_Images'
minhasPastas = os.listdir(pastaPrincipal)
imagens = []
def recorte(frame):
    #recorte topo
    if not np.sum(frame[0]):
        return recorte(frame[1:])
    #recorte embaixo
    if not np.sum(frame[-1]):
        return recorte(frame[:-2])
    #recorte lateral direita
    if not np.sum(frame[:,0]):
        return recorte(frame[:,1:])
    #recorte lateral esquerda
    if not np.sum(frame[:,-1]):
        return recorte(frame[:,:-2])
    return frame

def panoramica(img):
    iPanoramica = []
    for i in range (0,len(img)-1):
        if(i == 0):        
            img1 = cv2.cvtColor(img[i],cv2.COLOR_BGR2GRAY) #Passa para escala de cinza
            img2 = cv2.cvtColor(img[i+1],cv2.COLOR_BGR2GRAY) #Passa para escala de cinza
        else:
            img[i] = iPanoramica #Substitui a imagem já usada na iteração anterior pela panoramica atual
            img1 = cv2.cvtColor(img[i],cv2.COLOR_BGR2GRAY) #Passa para escala de cinza
            img[i+1] = cv2.resize(img[i+1],(0,0),None,0.8,0.8) #Redimensiona multiplicando a resolução atual pelo fator definido
            img2 = cv2.cvtColor(img[i+1],cv2.COLOR_BGR2GRAY) #Passa para escala de cinza
            
        sift = cv2.xfeatures2d.SIFT_create()
        # find the key points and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        match = cv2.FlannBasedMatcher(index_params, search_params)
        matches = match.knnMatch(des1,des2,k=2)

        good = []
        for m,n in matches:
            if m.distance < 0.5*n.distance:
                good.append(m)
        print(len(good))
        
        MIN_MATCH_COUNT = 10
        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            h,w = img1.shape
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            iPanoramica = cv2.perspectiveTransform(pts, M)
            img2 = cv2.polylines(img2,[np.int32(iPanoramica)],True,255,3, cv2.LINE_AA)
            iPanoramica = cv2.warpPerspective(img[i],M,(img[i+1].shape[1] + img[i].shape[1], img[i+1].shape[0]))
            iPanoramica[0:img[i+1].shape[0],0:img[i+1].shape[1]] = img[i+1]
        else:
            print("Não foram encontrados pontos suficientes - %d/%d", (len(good)/MIN_MATCH_COUNT))      
            return 0
    
    return recorte(iPanoramica)
        
imgTag = 1
for pasta in minhasPastas:
    local = pastaPrincipal +'/'+ pasta
    minhaLista = os.listdir(local)
    minhaLista.sort() 
    print(f"Numero total de imagens detectadas {len(minhaLista)}")
    for nomeImagem in minhaLista:
        imagemAtual = cv2.imread(f"{local}/{nomeImagem}")
        imagemAtual = cv2.resize(imagemAtual,(0,0),None,0.8,0.8) #O resize acontece para o processamento mais rapido
        imagens.append(imagemAtual)
        
    panoramicaR = panoramica(imagens)
    if (not(isinstance(panoramicaR, int))):
        cv2.imshow(f"panoramicaR{imgTag}", panoramicaR)
        cv2.imwrite(f"panoramicaR{imgTag}.jpg", panoramicaR)
        imgTag = imgTag + 1
        print(f"panoramicaR{imgTag} gerada")
        
    imagens = []
    

cv2.waitKey()

cv2.destroyAllWindows()


