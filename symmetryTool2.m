classdef symmetryTool2 < handle
    properties
        Figure
        HandleImage
        Image
        Overlay
        Dialog
        Slider
        Tform
        DoneButtonPushed
        Tx
        Ty
        Ta
        HalfIndex
        OutImage
        DisplayImage
        MidC
        SliderX
        SliderY
        SliderA
        SliderValueX
        SliderValueY
        SliderValueA
    end
    
    methods
        function tool = symmetryTool2(Image,varargin)
% similar to symmetryTool, but treats each vertical half independently
            tool.HalfIndex = 1;
            
            for i = 1:2
                tool.Tform{i} = affine2d(eye(3,3));
            end
            
            tool.DoneButtonPushed = false;
            
            tool.MidC = round(size(Image,2)/2);
            I = zeros(size(Image));
            I(:,1:tool.MidC) = Image(:,1:tool.MidC);
            tool.Image{1} = I;
            tool.OutImage{1} = I;
            I = zeros(size(Image));
            I(:,tool.MidC+1:end) = Image(:,tool.MidC+1:end);
            tool.Image{2} = I;
            tool.OutImage{2} = I;
            
            p = inputParser;
            p.addParameter('MaxShift',50);
            p.addParameter('MaxAngle',30);
            p.addParameter('MaxSink',50);
            p.parse(varargin{:});
            p = p.Results;
            maxShift = p.MaxShift;
            maxAngle = p.MaxAngle;
            
            ss = get(0,'ScreenSize'); % [left botton width height]
            tool.Figure = figure('Position',[ss(3)/4 ss(4)/4 ss(3)/2 ss(4)/2],...
                'NumberTitle','off', 'Name','Symmetry Tool', 'CloseRequestFcn',@tool.closeFigure, 'Resize','on');
            
            vPosition = size(Image,2)/2;
            
            tool.HandleImage = imshow(Image);
            hold on
            plot([vPosition vPosition], [1 size(Image,1)],'y')
            hold off
            
            dwidth = 400;
            dborder = 10;
            cwidth = dwidth-2*dborder;
            cheight = 20;
            
            tool.Dialog = dialog('WindowStyle', 'normal','Resize', 'on',...
                                'Name', 'SymTool',...
                                'CloseRequestFcn', @tool.closeDialog,...
                                'Position',[100 100 dwidth 6*dborder+6*cheight]);
            
            uicontrol('Parent',tool.Dialog,'Style','popupmenu','String',{'Left','Right'},'Position', [2*dborder 5*dborder+5*cheight cwidth-dborder cheight],'Callback',@tool.popupManage);
                            
            uicontrol('Parent',tool.Dialog,'Style','text','String','x','Position',[dborder 4*dborder+4*cheight 20 cheight],'HorizontalAlignment','left');
            tool.SliderX = uicontrol('Parent',tool.Dialog,'Style','slider','Min',-maxShift,'Max',maxShift,'Value',0,'Position',[2*dborder 4*dborder+4*cheight cwidth-dborder cheight],'Tag','dx');
            addlistener(tool.SliderX,'Value','PostSet',@tool.continuousSliderManage);
            
            uicontrol('Parent',tool.Dialog,'Style','text','String','y','Position',[dborder 3*dborder+3*cheight 20 cheight],'HorizontalAlignment','left');
            tool.SliderY = uicontrol('Parent',tool.Dialog,'Style','slider','Min',-maxShift,'Max',maxShift,'Value',0,'Position',[2*dborder 3*dborder+3*cheight cwidth-dborder cheight],'Tag','dy');
            addlistener(tool.SliderY,'Value','PostSet',@tool.continuousSliderManage);
            
            uicontrol('Parent',tool.Dialog,'Style','text','String','a','Position',[dborder 2*dborder+2*cheight 20 cheight],'HorizontalAlignment','left');
            tool.SliderA = uicontrol('Parent',tool.Dialog,'Style','slider','Min',-maxAngle,'Max',maxAngle,'Value',0,'Position',[2*dborder 2*dborder+2*cheight cwidth-dborder cheight],'Tag','da');
            addlistener(tool.SliderA,'Value','PostSet',@tool.continuousSliderManage);
            
            for i = 1:2
                tool.SliderValueX{i} = 0;
                tool.SliderValueY{i} = 0;
                tool.SliderValueA{i} = 0;
            end
            
            % quit
            uicontrol('Parent',tool.Dialog,'Style','pushbutton','String','Done','Position',[dborder dborder cwidth 2*cheight],'Callback',@tool.buttonDonePushed);
           
            for i = 1:2
                tool.Tx{i} = eye(3,3);
                tool.Ty{i} = eye(3,3);
                tool.Ta{i} = eye(3,3);
            end
            
            uiwait(tool.Dialog)
        end
        
        function popupManage(tool,src,~)
            tool.HalfIndex = src.Value;
            tool.SliderX.Value = tool.SliderValueX{tool.HalfIndex};
            tool.SliderY.Value = tool.SliderValueY{tool.HalfIndex};
            tool.SliderA.Value = tool.SliderValueA{tool.HalfIndex};
        end
        
        function continuousSliderManage(tool,~,callbackdata)
            value = callbackdata.AffectedObject.Value;
            tag = callbackdata.AffectedObject.Tag;
            
            i = tool.HalfIndex;
            tx = tool.Tx{i};
            ty = tool.Ty{i};
            ta = tool.Ta{i};
            
            if strcmp(tag,'dx')
                tx(3,1) = value;
                tool.SliderValueX{i} = value;
            elseif strcmp(tag,'dy')
                ty(3,2) = value;
                tool.SliderValueY{i} = value;
            elseif strcmp(tag,'da')
                a = -value/180*pi;
                tool.SliderValueA{i} = value;
                
                mp = fliplr(size(tool.Image{1})/2);
                T1 = [eye(2) [0; 0]; [mp 1]];
                r = [cos(a) -sin(a); sin(a) cos(a)];
                R = [[r [0; 0]]; [0 0 1]];
                T2 = [eye(2) [0; 0]; [-mp 1]];
                
                ta = T2*R*T1;
            end
            
            tool.Tx{i} = tx;
            tool.Ty{i} = ty;
            tool.Ta{i} = ta;
            
            tool.Tform{i} = affine2d(ta*ty*tx);
            tool.OutImage{i} = imwarp(tool.Image{i},tool.Tform{i},'OutputView',imref2d(size(tool.Image{i})));
            
            I1 = tool.OutImage{1};
            I2 = tool.OutImage{2};
            I = zeros(size(I1));
            I(:,1:tool.MidC) = I1(:,1:tool.MidC);
            I(:,tool.MidC+1:end) = I2(:,tool.MidC+1:end);
            
            tool.DisplayImage = I;
            tool.HandleImage.CData = tool.DisplayImage;
        end
        
        function I = applyTforms(tool,Image)
            I = zeros(size(Image));
            I(:,1:tool.MidC) = Image(:,1:tool.MidC);
            I1 = imwarp(I,tool.Tform{1},'OutputView',imref2d(size(I)));
            I = zeros(size(Image));
            I(:,tool.MidC+1:end) = Image(:,tool.MidC+1:end);
            I2 = imwarp(I,tool.Tform{2},'OutputView',imref2d(size(I)));
            I = zeros(size(Image));
            I(:,1:tool.MidC) = I1(:,1:tool.MidC);
            I(:,tool.MidC+1:end) = I2(:,tool.MidC+1:end);
        end
        
        function txy = applyTformsToSpots(tool,xy)
            xyLeft = xy(xy(:,1) <= tool.MidC,:);
            txyLeft = transformSpots(xyLeft,tool.Tform{1});
            xyRight = xy(xy(:,1) > tool.MidC,:);
            txyRight = transformSpots(xyRight,tool.Tform{2});
            txy = [txyLeft; txyRight];
        end
        
        function buttonDonePushed(tool,~,~)
            delete(tool.Figure);
            delete(tool.Dialog);
            tool.DoneButtonPushed = true;
        end
        
        function closeDialog(tool,~,~)
            delete(tool.Figure);
            delete(tool.Dialog);
        end
        
        function closeFigure(tool,~,~)
            delete(tool.Figure);
            delete(tool.Dialog);
        end
    end
    
    methods (Static)
        function I = staticApplyTforms(midC,tform1,tform2,Image)
            I = zeros(size(Image));
            I(:,1:midC) = Image(:,1:midC);
            I1 = imwarp(I,tform1,'OutputView',imref2d(size(I)));
            I = zeros(size(Image));
            I(:,midC+1:end) = Image(:,midC+1:end);
            I2 = imwarp(I,tform2,'OutputView',imref2d(size(I)));
            I = zeros(size(Image));
            I(:,1:midC) = I1(:,1:midC);
            I(:,midC+1:end) = I2(:,midC+1:end);
        end
        
        function txy = staticApplyTformsToSpots(midC,tform1,tform2,xy)
            xyLeft = xy(xy(:,1) <= midC,:);
            txyLeft = transformSpots(xyLeft,tform1);
            xyRight = xy(xy(:,1) > midC,:);
            txyRight = transformSpots(xyRight,tform2);
            txy = [txyLeft; txyRight];
        end
    end
end