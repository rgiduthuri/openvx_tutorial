/*
 * Copyright (c) 2016 The Khronos Group Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and/or associated documentation files (the
 * "Materials"), to deal in the Materials without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Materials, and to
 * permit persons to whom the Materials are furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Materials.
 *
 * THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.
 */

/*!
 * \file   opencv_camera_display.h
 * \brief  wrapper for OpenCV camera/file-input and display
 * \author Radhakrishna Giduthuri <radha.giduthuri@ieee.org>
 */

#ifndef __opencv_camera_display_h__
#define __opencv_camera_display_h__

#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <string>

#ifndef DEFAULT_VIDEO_SEQUENCE
#define DEFAULT_VIDEO_SEQUENCE "../../tutorial_videos/PETS09-S1-L1-View001.avi"
#endif

#ifndef DEFAULT_WAITKEY_DELAY
#define DEFAULT_WAITKEY_DELAY  1  /* waitKey delay time in milliseconds after each frame processing */
#endif

#ifndef ENABLE_DISPLAY
#define ENABLE_DISPLAY         1  /* display results using OpenCV GUI */
#endif

class CGuiModule
{
public:
    CGuiModule( const char * captureFile )
        : m_cap( captureFile ? captureFile : DEFAULT_VIDEO_SEQUENCE )
    {
        captureFile  = captureFile ? captureFile : DEFAULT_VIDEO_SEQUENCE;
        m_windowName = captureFile;
        if( !m_cap.isOpened())
        {
            printf( "ERROR: unable to open: %s\n", captureFile );
            exit( 1 );
        }
        printf( "OK: FILE %s %dx%d\n", captureFile, GetWidth(), GetHeight());
#if ENABLE_DISPLAY
        cv::namedWindow(m_windowName);
#endif
    }

    CGuiModule( int captureDevice )
        : m_cap( captureDevice )
    {
        char  name[64]; sprintf( name, "CAMERA#%d", captureDevice );
        m_windowName = name;
        if( !m_cap.isOpened())
        {
            printf( "ERROR: CAMERA#%d not available\n", captureDevice );
            exit( 1 );
        }
        printf( "OK: CAMERA#%d %dx%d\n", captureDevice, GetWidth(), GetHeight());
#if ENABLE_DISPLAY
        cv::namedWindow(m_windowName);
#endif
    }

    int GetWidth()
    {
        return (int) m_cap.get( cv::CAP_PROP_FRAME_WIDTH );
    }

    int GetHeight()
    {
#if 1 // TBD: workaround for reported OpenCV+Windows bug that returns width instead of height
		return 480;
#else
        return (int) m_cap.get( cv::CAP_PROP_FRAME_HEIGHT );
#endif
    }

    int GetStride()
    {
        return (int) m_imgRGB.step;
    }

    unsigned char * GetBuffer()
    {
        return m_imgRGB.data;
    }

    bool Grab()
    {
        m_cap >> m_imgBGR;
        if( m_imgBGR.empty() )
        {
            return false;
        }
        cv::cvtColor( m_imgBGR, m_imgRGB, cv::COLOR_BGR2RGB );
        return true;
    }

    void DrawText( int x, int y, const char * text )
    {
        cv::putText( m_imgBGR, text, cv::Point( x, y ),
                     cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar( 128, 0, 0 ), 1, cv::LineTypes::LINE_AA );
#if !ENABLE_DISPLAY
        printf("text: %s\n", text);
#endif
    }

    void DrawPoint( int x, int y )
    {
        cv::Point  center( x, y );
        cv::circle( m_imgBGR, center, 1, cv::Scalar( 0, 0, 255 ), 2 );
    }

    void DrawArrow( int x0, int y0, int x1, int y1 )
    {
        DrawPoint( x0, y0 );
        float  dx = (float) ( x1 - x0 ), dy = (float) ( y1 - y0 ), arrow_len = sqrtf( dx * dx + dy * dy );
        if(( arrow_len >= 3.0f ) && ( arrow_len <= 50.0f ) )
        {
            cv::Scalar  color   = cv::Scalar( 0, 255, 255 );
            float       tip_len = 5.0f + arrow_len * 0.1f, angle = atan2f( dy, dx );
            cv::line( m_imgBGR, cv::Point( x0, y0 ), cv::Point( x1, y1 ),                                                                                                              color, 1 );
            cv::line( m_imgBGR, cv::Point( x1, y1 ), cv::Point( x1 - (int) ( tip_len * cosf( angle + (float) CV_PI / 6 )), y1 - (int) ( tip_len * sinf( angle + (float) CV_PI / 6 ))), color, 1 );
            cv::line( m_imgBGR, cv::Point( x1, y1 ), cv::Point( x1 - (int) ( tip_len * cosf( angle - (float) CV_PI / 6 )), y1 - (int) ( tip_len * sinf( angle - (float) CV_PI / 6 ))), color, 1 );
        }
    }

    void Show()
    {
#if ENABLE_DISPLAY
        cv::imshow( m_windowName, m_imgBGR );
#endif
    }

    bool AbortRequested()
    {
        char  key = cv::waitKey( DEFAULT_WAITKEY_DELAY );
        if( key == ' ' )
        {
            key = cv::waitKey( 0 );
        }
        if(( key == 'q' ) || ( key == 'Q' ) || ( key == 27 ) /*ESC*/ )
        {
            return true;
        }
        return false;
    }

    void WaitForKey()
    {
#if ENABLE_DISPLAY
        cv::waitKey( 0 );
#endif
    }

protected:
    std::string       m_windowName;
    cv::VideoCapture  m_cap;
    cv::Mat           m_imgBGR;
    cv::Mat           m_imgRGB;
};

#endif
