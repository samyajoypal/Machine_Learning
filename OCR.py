# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 01:41:50 2019
@author: Samyajoy
"""


import io
import os
import re
import copy
import shutil
import subprocess
import tempfile

import concurrent.futures as ccf
import threading

import pandas as pd
import numpy as np

import pdf2image as pdfi
import PyPDF2 as pdf2
from PIL import Image,  ImageFilter
import pytesseract as pyt

import logging
log = logging.getLogger(__name__)


## Run OCR algorithms  #############################################################

def runOCR(pathFileNames, analyses, prepImages = False, lang = 'eng', sep = '\\'):
 
    msg = ' Run OCR analysis.\n'
    log.info(msg)
    print('\n', msg)

    if not(isinstance(pathFileNames, list)):
        raise IOError('\n Unable to OCR process files list %s!\n' %pathFileNames)
 
    alist_  = getList(analyses)
    if (len(alist_) == 0):
        msg = '\n Empty list of analyses to run, OCR finished.\n'
        log.info(msg)
        return
    
    index = 0
    for p_ in pathFileNames:
        if not(isinstance(p_, list)):
            raise IOError('\n Unable to OCR process files list %s!\n' %p_)

        ## Output directories
        path_ = p_[0] + sep + 'OCR'
        file_ = p_[1][0:p_[1].find('.', max(0,len(p_[1])-4))]
        try: 
            os.mkdir(path_, mode=2048)   ## mode = stat.S_ISUID
        except:
            pass
        
        alist__ = copy.deepcopy(alist_)

        msg = ' Run file %(ii)s of %(nn)s: %(f)s (%(p)s). ' %{'ii':index+1, 'nn':len(pathFileNames), 'f':p_[1], 'p':p_[0]}
        log.info(msg)
        print(msg)
        index += 1

        ## See whether files are already machine readable
        try:
            for ana_ in alist_:

                msg = '   PDF Processing of file %(f)s. ' %{'f':p_[1]}
                log.info(msg)
                # print(msg)

                a_ = ana_.lower()     
                if (a_=='txt'):
                    r = getFromPDF(p_[0], p_[1])
                    if (len(r.encode('utf-8').strip()) > 0):
                        writeTxtFile(r, path_, file_ + '_ocr.txt')
                        alist__.remove(ana_)  
                elif (((a_=='pdf') or (a_=='tsv') or (a_=='hocr')) and (p_[1].lower().endswith('.pdf'))):
                    r = getFromPDF(p_[0], p_[1], False, 0, 1)
                    if (len(r.encode('utf-8').strip()) > 0):
                        if (a_=='pdf'):
                            shutil.copy(p_[0] + sep + p_[1], path_ + sep + file_ + '_ocr.pdf')
                        else:
                            msg = '     tsv/hocr analysis not recommended for readable PDFs, please run manually.'                            
                            log.info(msg)
                        alist__.remove(ana_)
 
            if (len(alist__) == 0):
                msg = '   OCR for readable PDF finished.'
                log.info(msg)
                continue
        except Exception as err: 
            if not(p_[1].lower().endswith('.tif')):
                msg = '  --- Error: PDF Processing of file %(f)s failed.  -  %(ew)s --- ' %{'f':p_[1], 'ew':str(err)}
                log.error(msg)
                print(msg)

        pp_ = getPageCount(p_[0], p_[1])
        msg = '   Processing %(p)s pages for file %(f)s.' %{'p':pp_, 'f':p_[1]}
        log.info(msg)
        ## print(msg)
    
        ## threading text extractions
        threads = 15 
        batchsize = min(max(1, int(pp_/threads+0.5)), 20); 
        msg = '   Processing threaded batches of %s pages.' %batchsize
        log.info(msg)
        executor = ccf.ThreadPoolExecutor(max_workers=threads)
        threadList = list()
        rr_ = {'txt':list(), 'pdf':list(), 'hocr':list(), 'tsv':list()}
        ps_ = 0; pe_ = min(pp_, batchsize)
        while (ps_<pp_):        
            threadList.append(executor.submit(threadedTextExtraction, p_[0], p_[1], ps_, pe_, alist__, prepImages, lang))            
            ps_ = pe_; pe_ = min(pp_, pe_+batchsize)
        
        ## consolidate thread results
        for t_ in threadList:        
            rr__ = t_.result()
            for ana_ in alist__:
                for r_ in rr__[ana_]:
                    rr_[ana_].append(r_)
        
        ## save results        
        for ana_ in alist__:
            try:
                a_ = ana_.lower()               
                if (a_!='png'):
                    msg = '   Saving thread results.'
                    log.info(msg)
                    print(msg)

                if (a_=='txt'):
                    writeTxtFile(rr_[a_], path_, file_ + '_ocr.txt')
                elif (a_=='pdf'):
                    writePdfFile(rr_[a_], path_, file_ + '_ocr.pdf')
                elif (a_=="hocr"):
                    writeTxtFile(rr_[a_], path_, file_ + '_ocr.html')
                elif (a_=="tsv"):
                    writeTxtFile(rr_[a_], path_, file_ + '_ocr.tsv')
                elif (a_=='png'):
                    il_ = getImageList(p_[0], p_[1], 0, min(10, pp_))
                    for ip_, i_ in zip(range(len(il_)), il_):
                        f_ = path_ + sep + file_ + '_%04i.png' %ip_
                        log.debug('\t Saving temporary image %s.' %f_)
                        i_.save(f_)                
                    
            except Exception as err:
                msg = '  --- Error: Saving results failed for file %s!  -  %s  ---' %(p_[1], str(err))
                log.error(msg)
                print(msg)
    
    
        msg = '   File %s finished!\n' %p_[1] 
        log.info(msg)
        print('\n', msg)

    print('\n OCR finished.\n')
    
    
def threadedTextExtraction(path, file, startpage_, endpage_, analyses, prepImages = False, lang = 'eng', sep = '\\'):

    ## Get page images from files        
    il_ = getImageList(path, file, startpage_, endpage_)

    rr__ = {'txt':list(), 'pdf':list(), 'hocr':list(), 'tsv':list()}

    ## Apply preprocessing 
    if prepImages:
        il__ = list()
        page_ = startpage_
        for ii_ in range(len(il_)):

            page_ += 1
            msg = '   Start preprocessing page %s.' %page_
            log.info(msg)
            ## print(msg)
            
            actions = list()
            tImage = il_[ii_].copy()
            prepImage(tImage, actions)

            for a_ in actions:
                msg = '   Applying preprocessing: %(aa)s with parameter(s) %(pp)a for file %(f)s.' %{'aa':str(a_[0]).rsplit(' ')[1], 'pp':a_[1], 'f':file}
                log.info(msg)
                tImage = a_[0](tImage, a_[1])

            il__.append(copy.deepcopy(tImage))

        il_ = copy.deepcopy(il__)

    ## run OCR on page images        
    for ana_ in analyses:
        try:
            a_ = ana_.lower()               
            if (a_!='png'):
                msg = '   Running Tesseract %s analysis on pages %s to %s.' %(a_, startpage_, endpage_)
                log.info(msg)
                print(msg)
                r_ = getFromImages(il_, a_, lang)
                page_ = startpage_
                for r__ in r_:
                    if (a_ in ['txt', 'tsv']):
                        r__ = '\n\n\n --- Tesseract results for page ' + str(page_+1) + '. --- \n\n' + r__
                        page_+=1
                    rr__[a_].append(r__)
            
        except Exception as err:    
            msg = '  --- Error: Tesseract failed for file %(ff)s!  -  %(ew)s  ---' %{'ff':file, 'ew':str(err)}
            log.error(msg)
            print(msg)

    return rr__


def runRegEx(pathFileNames, regExList, wholeSection = True, outPath = '', sep = '\\'):
 
    if not(isinstance(pathFileNames, list)):
        raise IOError('Unable to OCR process files list %s!' %pathFileNames)

    msg = ' Running %(n)s reg. expression(s) analyses for %(f)s files.' %{'n':len(regExList), 'f':len(pathFileNames)}
    ## log.info('\n', msg)
    print(msg)
 
    result_ = list()
    
    ## For each fiel run all the regular expression searches
    ## wholesection = True means the whole enclosing paragraph is returned 
    ##   defined by two enclosing blank lines
    for p_ in pathFileNames:
        if not(isinstance(p_, list)):
            raise IOError('Unable to process files list %s!' %p_) 

        r_ = list()
        for re_ in getList(regExList):
            try:
                regex_ = re.compile(re_)
                rsect_ = re.compile("(^[ \t]*$)")
                rpage_ = re.compile("( --- Tesseract results for page)")
                buffer_ = list()
                r__ = list()
                found_ = False
                blen_ = 0
                page_ = ''
                
                f_ = open(p_[0] + sep + p_[1], "r", encoding='utf-8-sig')
                for l_ in f_:
                    if (rpage_.search(l_)):
                        page_ = l_
                    if (wholeSection):
                        if (rsect_.search(l_)):
                            if (found_):
                                buffer_ = buffer_[max(0,blen_-4):min(blen_+3, len(buffer_))]
                                buffer_.append('\n')
                                r__.append(page_)
                                r__.append(copy.deepcopy(buffer_))
                                found_ = False
                            buffer_.clear()
                        else:
                            buffer_.append(l_)
        
                        if (regex_.search(l_)):
                            found_ = True
                            blen_ = len(buffer_)
                    else:
                        if (regex_.search(l_)):
                            r__.append(copy.deepcopy(l_))
                            ##r__.append('\n')
                        
                r_.append(r__)                    
                f_.close()
            except Exception as err: 
                msg = '  --- Error: RegEx analysis failed for file %(f)s and reg. ex. %(r)s!  -  %(ew)s  ---' %{'f':p_[1], 'r':re_, 'ew':str(err)}
                ## log.error(msg)
                print(msg)

                
        result_.append(r_)                    
 
    ## Save results in oine common file or in several individual file is in the source directories
    if (outPath != ''):
        of_ = open(outPath + sep + "RegExAnalysis.txt", "w")               
    for r_, p_ in zip(result_, pathFileNames):

        if (outPath == ''):
            of_ = open(p_[1][0:p_[1].find('.')] + "_rex.txt", "w", encoding='utf-8-sig')
        else:
            of_.write("File name: \t" + p_[0] + sep + p_[1] + "\n\n")
        
        for r__, rx_ in zip(r_, getList(regExList)):
            try:
                of_.write("\t Regular expression:\t" + rx_ + "\n\n")
                for l_ in r__:
                    if isinstance(l_, list):
                        for l__ in l_:
                            of_.write(l__)
                    else: 
                        print(l_)
                        of_.write(l_)
                of_.write("\n")
            except Exception as err: 
                of_.write("\n\t Can\'t write regular expression "+ rx_ + " results for " +  p_[0] + sep + p_[1] + ".  -  %s  \n\n" %str(err))
            of_.write("\n")
            
            
        if (outPath == ''):
            of_.close()

    if (outPath != ''):
        of_.close()

    msg = '     RegExAnalysis finished.'
    log.info(msg)
    print(msg)
 
    
def getRegExResults(path, filename, sep='\\'):
    reResults = pd.DataFrame(columns='File,RegEx,Text'.split(','))
    f = open(path+sep+filename,'r')
    fname = ''; rename = ''; reText = ''
    for l in f:
        if (l.strip(' \t').startswith('File name:')):
            if (fname != ''):
                reResults.loc[reResults.shape[0]] = [fname, rename, reText]   
                rename = ''
                reText = ''
            fname = l.strip(' \t')[10:len(l.strip(' \t'))].strip(' \t')
        elif (l.strip(' \t').startswith('Regular expression:')):
            if (fname != '' and rename != ''):
                reResults.loc[reResults.shape[0]] = [fname, rename, reText]   
                rename = ''
                reText = ''
            rename = l.strip(' \t')[19:len(l.strip(' \t'))].strip(' \t')
        elif (l.strip() != ''):
            reText += l.strip(' \t')
        
    return reResults
    
    
## Extract from PDFs  ##############################################################
    
def getPageCount(path, filename, sep = '\\'):
    try:
        file_  = path + sep + filename
        if (filename.lower().endswith('.tif')):
        ## Load tif pages
            images_ = Image.open(file_)
            try:
                return images_.n_frames
            except:
                pass
            i_ = 0
            while True:
                try:
                    images_.seek(i_)
                    i_ = i_+1
                except:
                     break
            return i_
        else:
        ## Load pdf pages 
            return countPdfPages(file_)
            
    except Exception as err:
        msg = '  --- Error: Unable to assert page count of file %s!  -  %s  ---' %(filename, str(err))
        log.error(msg)
        print(msg)
        

    
def getImageList(path, filename, startpage = 0, endpage = 0, sep = '\\'):
    
    lastpage = getPageCount(path, filename)
    startpage_ = max(0, startpage)
    
    if (endpage == 0):
        endpage_ = lastpage
    else:
        endpage_ = max(startpage_, min(lastpage, endpage))
    
    if (startpage_>=endpage_):
        msg = 'Startpage %s >= endpage %s for file %s!' %(startpage, endpage, filename)
        log.info(msg)
        print(msg)
        return list()
    
    try:
        file_  = path + sep + filename
        if (filename.lower().endswith('.tif')):
        ## Load tif pages
            imageList_ = list()
            images_ = Image.open(file_)
            i_ = startpage_
            while (i_ < endpage_): #for i_ in range(images_.n_frames):
                try:
                    images_.seek(i_)
                    imageList_.append(images_.copy())
                    i_ = i_+1
                except:
                     break
            return imageList_
        else:
        ## Load pdf pages 
            try:
                path = tempfile.TemporaryDirectory()
                imageList_ = copy.deepcopy(pdfi.convert_from_path(file_, dpi = 200, output_folder=path.name, first_page=(startpage_+1), last_page=endpage_, thread_count=4))
                path.cleanup()
                return imageList_
            except:
                pass
            path = tempfile.TemporaryDirectory()
            imageList_ = copy.deepcopy(pdfi.convert_from_path(file_, dpi = 100, output_folder=path.name, first_page=startpage_+1, last_page=endpage_, thread_count=4))
            path.cleanup()
            return imageList_
    except Exception as err:
        msg = '  --- Error: Unable to get image pages %s to %s for file %s!  -  %s  ---' %(startpage_, endpage_, filename, str(err))
        log.error(msg)
        print(msg)
        return list()


def getFromImages(pages, outformat = 'txt', lang = 'eng'):
    
    pages_ = list()
    pageList_ = list()
        
    if not(isinstance(pages, list)):
        pages_.append(pages)
    else:
        pages_ = pages

    msg = '     Running Tesseract %(of)s analysis on %(np)s page(s). ' %{'of':outformat, 'np':len(pages_)}
    log.debug(msg)
    ## print(msg)

    for page_ in pages_:
        try:
            if (outformat == 'txt'):
                configStr_ = '-l ' + lang + ' --psm 3' ## , tessedit_write_images=1
                pageList_.append(pyt.image_to_string(page_, config = configStr_))
            elif (outformat == 'pdf'):
                configStr_ = '-l ' + lang + ' --psm 3 -c tessedit_create_pdf=1'
                pageList_.append(pyt.image_to_pdf_or_hocr(page_, config = configStr_, extension='pdf'))
            elif (outformat == 'hocr'):
                configStr_ = '-l ' + lang + ' --psm 3 -c tessedit_create_hocr=1'
                pageList_.append(pyt.image_to_pdf_or_hocr(page_, config = configStr_, extension='hocr').decode("utf-8"))
            elif (outformat == 'tsv'):
                configStr_ = '-l ' + lang + ' --psm 3 -c tessedit_create_tsv=1'
                pageList_.append(pyt.image_to_pdf_or_hocr(page_, config = configStr_, extension='tsv').decode("utf-8"))
        except Exception as err:
            msg = '  --- Error: Unable to OCR process image data!  -  %s  ---' %str(err)
            log.error(msg)
            ## print(msg)
        

    return pageList_
  

def getFromPDF(path, filename, pageNums = False, startPage = 0, endPage = 0, sep = '\\'):
 
    pages = countPdfPages(path + sep + filename)
    if ((endPage>startPage)):
        pages = endPage

    msg = '   PDF Processing %(p)s page(s) for file %(f)s. ' %{'p':pages-max(0, startPage), 'f':filename}
    log.debug(msg)
    ## print(msg)
        
    ofile = tempfile.mktemp(prefix='ocr_')
    ## print(ofile)

    text =  ''
    if (not(pageNums) and (endPage == 0)):
        ## Call xpdf
        subprocess.call(['pdftotext', '-layout', path + sep + filename, ofile])
        ## Opens file saved to disk 
        file = open(ofile,'r', encoding = "ISO-8859-1")
        text = file.read()
        file.close()
   
    else:
        if pages == 0:
            pages = 1000
        for i in range(max(0, startPage), pages):
            actual = i + 1
            try:
                ## Calls xpdf 
                subprocess.call(['pdftotext', '-f', str(actual),'-l', str(actual), path + sep + filename, ofile])
                # Opens file saved to disk 
                file = open(ofile,'r', encoding = "ISO-8859-1") ## "ISO-8859-1" "Windows-1252"
                text = file.read()
                ## If the page is blank, it is not a real page
                if text == '':
                    continue
                ## Add text and page count to existing string
                if (pageNums):
                    text += '\n\n  ***Page {}*** \n\n {}'.format(actual, i)
                file.close()

            except:
                continue
            
	 ## Remove file saved to disk
    try:
        os.remove(ofile)
    except OSError:
        pass       

    return text


def countPdfPages(filename):  
    try:     
        pdf_ = pdf2.PdfFileReader(filename)
        return pdf_.getNumPages()
    except:
        try:
            rxcountpages = re.compile(r"/Type\s*/Page([^s]|$)", re.MULTILINE|re.DOTALL)
            data = open(filename,"r", encoding = "ISO-8859-1").read()
            return len(rxcountpages.findall(data))
        except:
            return 0


def findFiles(path, suffix, recursive = False, exclude = list(['OCR']), sep = '\\'):
    files_ = list()    
    for i in os.scandir(path):
        if ((i.is_dir() and recursive)):
            files__ = findFiles(path + sep + i.name, suffix, recursive, exclude, sep)
            for f_ in files__:
                files_.append(f_)
        elif (i.is_file()):
            
            in_ = False
            for s_ in getList(suffix):
                if (i.name.lower().endswith(s_)):
                    in_ = True
            for e_ in getList(exclude):
                path_ = path
                if (path.find('Library'+sep+'Python'+sep+'OCR') > 0):
                    path_ = path[path.find('Library'+sep+'Python'+sep+'OCR')+18:len(path)]
                if not((path_ + i.name).find(e_)<0):                    
                    in_ = False

            if (in_):
                files_.append([path, i.name])

    return files_


def writeTxtFile(result, path, filename, sep = '\\'):
    textFile_ = open(path + sep + filename, "w", encoding = 'utf-8')
    if (isinstance(result, list)):
        for lr_ in result:
            textFile_.write(lr_)
    else:
        textFile_.write(result)
    textFile_.close()
    

def writePdfFile(result, path, filename, sep = '\\'):
    pdfFile_ = open(path + sep + filename, "wb")
    pdfw = pdf2.PdfFileWriter()
    if (isinstance(result, list)):
        for lr_ in result:
            ios = io.BytesIO()
            ios.write(lr_)
            pdf_ = pdf2.PdfFileReader(ios)
            pdfw.addPage(copy.deepcopy(pdf_.getPage(0)))
            ios.close()
    else:
            ios = io.BytesIO()
            ios.write(result)
            pdf_ = pdf2.PdfFileReader(ios)
            pdfw.addPage(copy.deepcopy(pdf_.getPage(0)))
            ios.close()            
    pdfw.write(pdfFile_)
    pdfFile_.close()
    

def getList(myArg):
    l_ = list()
    if (isinstance(myArg, list)):
        l_ = myArg
    elif (isinstance(myArg, str)):
        l_.append(myArg)
    else:
        raise IOError('List or string required!') 
    return l_


## Helper  #########################################################################

def rotateImage(img, angle):
    return img.rotate(angle, expand=True)


def maxFilter(img, win):
    return img.filter(ImageFilter.MaxFilter(win))

def minFilter(img, win):
    return img.filter(ImageFilter.MinFilter(win))

def rankFilter(img, win):
    return img.filter(ImageFilter.RankFilter(win, int(win*win/4*3)))

def gaussianBlur(img, win):
    return img.filter(ImageFilter.GaussianBlur(win))

def boxBlur(img, win):
    return img.filter(ImageFilter.BoxBlur(win))

def unsharpMask(img, win):
    return img.filter(ImageFilter.UnsharpMask(win))


def threshold(img, t):

    def pixelThres(intensity):
        if ((intensity>t[1]) or (intensity<t[0])):
            return intensity
        else:
            return 255
    
    multiBands   = img.convert('RGB').split()
    redBand      = multiBands[0].point(pixelThres)
    greenBand    = multiBands[1].point(pixelThres)
    blueBand     = multiBands[2].point(pixelThres)
             
    #redBand.show()
    #greenBand.show()
    #blueBand.show()
    
    return Image.merge("RGB", (redBand, greenBand, blueBand))


prepOptions = {
        'maxFilter':    maxFilter,
        'minFilter':    minFilter,
        'rankFilter':   rankFilter,
        'rotate':       rotateImage,
        'threshold':    threshold,
        'gaussianBlur': gaussianBlur,
        'boxBlur':      boxBlur,
        'unsharpMask':  unsharpMask
        }


def runCrop(img):
    tsvi_ = getFromImages(img, 'tsv', 'eng') # nld
    tsv_  = pd.read_csv(io.StringIO(tsvi_[0]), encoding='ISO-8859-1', delimiter='\t', quoting=3, # csv.QUOTE_NONE,
                  header=0, names=['Level', 'Page', 'Block', 'Paragraph', 'Line', 
                                   'Word', 'Left', 'Top', 'Width', 'Height', 'Confidence', 'Text'] #, dtype = {'Level': int, 'Page': int, 'Block': int, 'Paragraph': int, 'Line': int, 'Word': int, 'Left': int, 'Top': int, 'Width': int, 'Height': int, 'Confidence': int, 'Text': str})
                  )
    try:
        tsv_ = tsv_[tsv_['Level'].apply(lambda l: l != 'level')]  
        tsv_['Confidence'] = pd.to_numeric(tsv_['Confidence'])
        tsv_ = tsv_[tsv_['Confidence'].apply(lambda c: c != -1)]  
        tsv_ = copy.deepcopy(tsv_[tsv_['Text'].apply( lambda t: len(str(t).encode('utf-8').strip()) > 2)])
        log.debug('     Tesseract tsv output: ' + str(list(tsv_['Text'])))    
        return np.average(tsv_['Confidence']), np.count_nonzero(tsv_['Confidence'])
    except Exception as err: 
        log.debug('  --- Error running crop OCR.  - %s  ---  ' %str(err))
        pass

    return 0, 0


 ## Crop part of the image and try to optimse preprocessing
def prepImage(img, actions):
    
    img_ = img.copy()

    is_ = img.size
    log.debug('     Page size: \t %(is)s' %{'is':is_})
    
    lo_ = [0.1, 0.175, 0.25, 0.325, 0.4]
    bs_ = 0.25
    as_ = [0, -90, 90, 180]

    counts_ = 0; confi_ = 0
    af_ = 0; cs_ = 0; cf_ = 0
    bb_ = [0, 0, 0, 0]
    bbox_ = [0, 0, 0, 0]
    for aa in as_:
        for ll in lo_:

            l_ = int( 0.1 * is_[0])
            r_ = int( ll * is_[1])
            t_ = int( 0.9 * is_[0])
            b_ = int((ll + bs_) * is_[1])
            if ((aa < -45 and aa > -135) or (aa > 45 and aa < 135) or (aa > 225 and aa < 315)):
                l_ = int( ll * is_[0])
                r_ = int( 0.1 * is_[1])
                t_ = int((ll + bs_) * is_[0])
                b_ = int( 0.9 * is_[1])
            bb_ = [l_, r_, t_, b_]

            ## print(bb_)
            
            try:
                img__ = img.crop(box=bb_).copy()
                img__ = rotateImage(img__, aa)
                # img__.show()
                cf_, cs_ = runCrop(img__)
                log.debug('     Box %(bb)s, Angle %(as)s, Counts %(cs)s, Confidence %(cf)s' %{'bb':bb_, 'as': aa, 'cs':cs_, 'cf': cf_})
            except Exception as err:
                log.debug('     Running box OCR failed for angle %(aa)s and box %(bb)s.  -  %(ew)  ' %{'aa':aa, 'bb':bb_, 'ew':str(err)} )    
                pass
            
            if ((((counts_ < 25) or (cs_ > counts_)) and (cf_ > 30)) or 
                (((cs_ >= 25)) and (cf_ > 50) and (cf_ > confi_))):
                counts_ = cs_
                confi_ = cf_
                bbox_ = bb_
                if (aa != 0 ):
                    af_ = aa
                img_ = img__.copy()
               
        if (counts_ >= 25 and confi_ >= 50):  
            break;


    ## If we can't find reasonable region run hole image
    if (counts_ < 25 or confi_ < 50):
        counts_ = 0; confi_ = 0
        af_ = 0
        bb_ = [0,0,is_[0],is_[1]]        
        img_ = img.copy()
        for aa in as_:

            try:
                img__ = rotateImage(img_, aa)
                # img__.show()
                cf_, cs_ = runCrop(img__)
                log.debug('     Box %(bb)s, Angle %(as)s, Counts %(cs)s, Confidence %(cf)s' %{'bb':bb_, 'as': aa, 'cs':cs_, 'cf': cf_})
            except Exception as err:
                log.debug('     Running box OCR failed for angle %(aa)s and box %(bb)s.  -  %(ew)  ' %{'aa':aa, 'bb':bb_, 'ew':str(err)} )    
                pass
            
            if (((cs_ > counts_) and (cf_ > 30)) or 
                ((cs_ >= 50) and (cf_ > confi_))):
                counts_ = cs_
                confi_ = cf_
                bbox_ = bb_
                if (aa != 0 ):
                    af_ = aa
                img_ = img__.copy()
               

    img_.show()
    if (counts_ >= 25 and confi_ > 90):
        if (af_ != 0):
            actions.append((prepOptions['rotate'], af_))
        log.info('     Preprocessing --- Final box %(bb)s, Angle %(af)s, Counts %(cs)s, Confidence %(cf)s ' %{'bb': bbox_, 'af': af_, 'cs':counts_, 'cf': confi_})
        return
    else:
        log.debug('     Final box %(bb)s, Angle %(af)s, Counts %(cs)s, Confidence %(cf)s ' %{'bb': bbox_, 'af': af_, 'cs':counts_, 'cf': confi_})
        
 
    ## Find best angle
    angles_ = [-2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2]
    img__ = img_.copy()
    aa_ = 0
    for a_ in angles_:
        try:
            img__ = rotateImage(img_, a_)
            cf_, cs_ = runCrop(img__)
            #img__.show()
            log.debug('     Angle %(angle)s, Counts %(cs)s, Confidence %(cf)s' %{'angle': a_, 'cs':cs_, 'cf': cf_})
        except:
            pass

        if (((cs_ > 0.75*counts_) and (cf_ > confi_)) or 
            ((cs_ >= 10) and (cf_ > 50) and (cf_ > confi_))):
            counts_ = cs_
            confi_ = cf_
            aa_ = a_
            
    if (af_+aa_ != 0):
        actions.append((prepOptions['rotate'], af_+aa_))
        img_ = rotateImage(img_, aa_)        
        # img_.show()
        
    ## img_.show()
    if (counts_ >= 25 and confi_ > 90):
        log.info('     Preprocessing --- Final box %(bb)s, Angle %(af)s, Counts %(cs)s, Confidence %(cf)s ' %{'bb': bbox_, 'af': af_+aa_, 'cs':counts_, 'cf': confi_})
        return
    else:
        log.debug('     Final box %(bb)s, Angle %(af)s, Counts %(cs)s, Confidence %(cf)s ' %{'bb': bbox_, 'af': af_+aa_, 'cs':counts_, 'cf': confi_})
   
    
    ## Find the best combination of some filtering   
    filter_ = ['maxFilter', 'minFilter', 'rankFilter', 'gaussianBlur', 'boxBlur']
    window_ = [3, 5]   
    thres_  = [[50, 250], [100, 250], [50, 200], [100, 200]]
    img_local = img_
    for i_ in range(5):
        
        ff_ = ''
        ww_ = 1    
        
        for t_ in thres_:
            try:
                img__ = threshold(img_local, t_)
                ## img__.show()
                cf_, cs_ = runCrop(img__)
                log.debug('     Threshold - levels %(t)s: Counts %(cs)s, Confidence %(cf)s' %{'t':t_, 'cs':cs_, 'cf': cf_})
            except:
                pass
    
            if ((cs_ >= 0.5*counts_) and (cf_ > 50) and (cf_ > confi_)): # 0.75*counts_)):
                counts_ = cs_
                confi_ = cf_
                ff_ = 'threshold'
                ww_ = t_
        
        
        for f_ in filter_:
            for w_ in window_:
                try:
                    img__ = prepOptions[f_](img_local, w_)
                    ## img__.show()
                    cf_, cs_ = runCrop(img__)
                    log.debug('     Filter %(f)s - window %(w)s: Counts %(cs)s, Confidence %(cf)s' %{'f':f_, 'w':w_, 'cs':cs_, 'cf': cf_})
                except:
                    pass
        
                if ((cs_ >= 0.5*counts_) and (cf_ > 50) and (cf_ > confi_)): # 0.75*counts_)):
                    counts_ = cs_
                    confi_ = cf_
                    ff_ = f_
                    ww_ = w_
        
        if (len(ff_) > 0):
            actions.append((prepOptions[ff_], ww_))
            img_local = prepOptions[ff_](img_local, ww_)
        else:
            break
   
    log.info('     Preprocessing --- Final box %(bb)s, Angle %(af)s, Counts %(cs)s, Confidence %(cf)s ' %{'bb': bbox_, 'af': af_+aa_, 'cs':counts_, 'cf': confi_})
           
    
## Remove watermarks ###############################################################

def removeWatermark(path, file, wm, replcmnt = '', rplc=False, sep ='\\'):

    print('Trying to remove watermark from ', path + sep + file, '.')
    wm_ = getList(wm)
    ifile_ = path + sep + file
    ist_ = open(ifile_,'rb')
    try:
        pdf_ = pdf2.PdfFileReader(ist_)
        out_ = pdf2.PdfFileWriter()

        for p_ in range(15): ##pdf_.getNumPages()):
            if (int(p_/20) == 1.0*p_/20.0): 
                print('Running page %s of %s.' %(p_+1, pdf_.getNumPages()))
            page_= pdf_.getPage(p_)
            co = page_['/Contents'].getObject()
            cs = pdf2.pdf.ContentStream(co, p_)
        
            for operands, operator in cs.operations:
                if (operator == pdf2.utils.b_("Tj")) or (operator == pdf2.utils.b_("TJ")) or (operator == pdf2.utils.b_("cm")) or (operator == pdf2.utils.b_("CM")):
                    text = operands[0]
                    for w_ in wm_:
                        
                            try:
                                if (isinstance(text, pdf2.generic.TextStringObject) and (text.find(w_) >= 0)):  
                                    ## print('vorher \t', operands[0])
                                    operands[0] = pdf2.generic.TextStringObject(replcmnt)
                                    page_.__setitem__(pdf2.generic.NameObject('/Contents'), cs)
                                    ## print('nachher \t', operands[0])
                                
                                else: 
                                    t__ = l2s(text)
                                    if ((t__.find(w_) >= 0)): ## isinstance(text[0], pdf2.generic.TextStringObject) and 
                                        ## print('vorher \t', t__)
                                        operands[0] = o2r(operands[0], pdf2.generic.TextStringObject(replcmnt))
                                        page_.__setitem__(pdf2.generic.NameObject('/Contents'), cs)   
                                        ## print('nachher \t', operands[0])
                            except:
                                ## print('Errors occurring for page %s.' %(no__+1))
                                pass
            
            out_.addPage(page_)
                
            
        try:
            ofile_ = path + sep + file[0:file.find('.', len(file)-4)] + '_wowm.pdf'
            print('Writing file ', ofile_, '.')
            ost_ = open(ofile_,'wb')
            out_.write(ost_)    
            ost_.close()
            ist_.close()
            if (rplc):
                shutil.move(ifile_, path + sep + file[0:file.find('.', len(file)-4)] + '_orig.pdf')
                shutil.move(ofile_, ifile_)
            print('Finished removal of watermark from ', path + sep + file, '.\n')
        except:
            raise IOError('Can\'t write file ', ofile_)
            
    except:
        raise IOError('Can\'t remove watermark from ', path + sep + file)


def l2s(list__):
    if not(isinstance(list__, list)):
        return list__
    index = 0
    r__ = ''
    while (index<len(list__)):
        r__ += list__[index]
        index += 2
    return r__
   
def o2r(op__, rc__):
    index = 0
    r__ = op__
    while (index<len(op__)):
        r__[index] = rc__
        index += 2
    return r__


def findWatermark(path, file, wm, sep ='\\'):

    wm_ = getList(wm)
    ifile_ = path + sep + file
    ist_ = open(ifile_,'rb')
    try:
        pdf_ = pdf2.PdfFileReader(ist_)
        for p_ in range(min(2,pdf_.getNumPages())):
            page_=pdf_.getPage(p_)  
            co = page_['/Contents'].getObject()
            cs = pdf2.pdf.ContentStream(co, p_)
            for operands, operator in cs.operations:
                if (operator == pdf2.utils.b_("Tj")) or (operator == pdf2.utils.b_("TJ")) or (operator == pdf2.utils.b_("cm")) or (operator == pdf2.utils.b_("CM")):
                    text = operands[0]
                    for w_ in wm_:
                        try:
                            if (isinstance(text, pdf2.generic.TextStringObject) and (text.find(w_) >= 0)): ## isinstance(text, pdf2.generic.TextStringObject) and 
                                return path, file, True
                        except:
                            try:
                                t__ = l2s(text)
                                if ((t__.find(w_) >= 0)): ## isinstance(text[0], pdf2.generic.TextStringObject) and 
                                    return path, file, True
                            except:
                                pass
    except:
        return path, file, IOError("Error!")

    return path, file, False

    
## #################################################################################
