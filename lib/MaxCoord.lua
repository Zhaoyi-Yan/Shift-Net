--[[
Expects 3D or 4D input. Does a max over the feature channels.
--]]
local MaxCoord, parent = torch.class('nn.MaxCoord', 'nn.Module')

function MaxCoord:__init(inplace)
    parent.__init(self)
    self.inplace = inplace or false

end

function MaxCoord:updateOutput(input)
    local nInputDim = input:nDimension()
    if input:nDimension() == 3 then
        local C,H,W = input:size(1), input:size(2), input:size(3)
        input = input:view(1,C,H,W)
    end
    assert(input:nDimension()==4, 'Input must be 3D or 4D (batch).')
    if self._type ~= 'torch.FloatTensor' then
        input = input:float()
    end
    input = input:float()


    local _, argmax = torch.max(input,2)  

    if self.inplace then
        self.output = input:zero()
    else
        self.output = torch.FloatTensor():resizeAs(input):zero()
    end
    local N = input:size(1)
    -- self.ind must be a tensor of 1 ndimension
    local h = self.output:size(3)
    local w = self.output:size(4)
    local spSize = h*w
    self.ind = torch.Tensor(N*spSize):zero()
    local tmp, ind_tmp
    for b=1,N do
        for i=1,h do
            for j=1,w do
                ind_tmp = argmax[{b,1,i,j}]
                self.output[{b,ind_tmp,i,j}] = 1

                tmp = (b-1)*spSize + (i-1)*w+j
                self.ind[tmp] = ind_tmp
            end
        end
    end

    self.output = self.output:type(self._type)

    if nInputDim == 3 then
        self.output = self.output[1]
    end

    return self.output, self.ind
end


function MaxCoord:updateGradInput(input, gradOutput)
    -- do nothing
    assert(1==2)
    return self.gradInput
end