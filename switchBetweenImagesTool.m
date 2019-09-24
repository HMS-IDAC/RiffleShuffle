classdef switchBetweenImagesTool < handle
    properties
        Figure
        Axis
        Handle
        Image
%         Dialog
        ShowingFirstImage
    end
    
    methods
        function tool = switchBetweenImagesTool(Image1,Image2)
            tool.Image{1} = Image1;
            tool.Image{2} = Image2;
            
            tool.Figure = figure('NumberTitle','off', 'Name','Image 1 (press Space to switch, Esc to quit)', 'CloseRequestFcn',@tool.closeTool,...
                                 'Resize','on', 'KeyPressFcn',@tool.keyPressed);
            
            tool.Axis = axes('Parent',tool.Figure,'Position',[0 0 1 1]);
            tool.Handle = imshow(tool.Image{1});
            tool.ShowingFirstImage = true;
            
%             dwidth = 200;
%             dborder = 10;
%             cwidth = dwidth-2*dborder;
%             cheight = 20;
%             
%             tool.Dialog = dialog('WindowStyle', 'normal',...
%                                 'Name', 'SBI',...
%                                 'CloseRequestFcn', @tool.closeTool,...
%                                 'Position',[100 100 dwidth 2*dborder+2*cheight],...
%                                 'KeyPressFcn',@tool.keyPressed);
%             
%             % switch
%             uicontrol('Parent',tool.Dialog,'Style','pushbutton','String','Switch [click here or press Space]','Position',[dborder dborder cwidth 2*cheight],'Callback',@tool.buttonSwitchPushed);
           
            uiwait(tool.Figure)
        end
        
        function keyPressed(tool,~,event)
            if strcmp(event.Key,'space')
                tool.buttonSwitchPushed(tool);
            elseif strcmp(event.Key,'escape')
                tool.closeTool(tool);
            end
        end
        
        function buttonSwitchPushed(tool,~,~)
            if tool.ShowingFirstImage
                tool.Handle.CData = tool.Image{2};
                tool.ShowingFirstImage = false;
                tool.Figure.Name = 'Image 2 (press Space to switch, Esc to quit)';
            else
                tool.Handle.CData = tool.Image{1};
                tool.ShowingFirstImage = true;
                tool.Figure.Name = 'Image 1 (press Space to switch, Esc to quit)';
            end
        end
        
        function closeTool(tool,~,~)
            delete(tool.Figure);
%             delete(tool.Dialog);
        end
        
    end
    
end
