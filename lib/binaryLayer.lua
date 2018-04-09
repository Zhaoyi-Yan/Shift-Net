local BinaryLayer, parent = torch.class('nn.BinaryLayer', 'nn.Module')

function BinaryLayer:__init(cls, threshold)
    parent.__init(self)
    self.cls = cls
    self.threshold = threshold
end

-- It abandoms the last channel(mask)
function BinaryLayer:updateOutput(input)
    assert(input:nDimension() == 4)
    self.bz = input:size(1)
    assert(self.bz == 1)
    self.c = input:size(2)
    self.h = input:size(3)
    self.w = input:size(4)
    self.real_c = self.c - 1

    if self.c == 1 then
        local input_float = input:squeeze():float()
        local mask_bin = binary_mask(input_float, self.threshold)
        self.ex_mask = mask_bin
        self.output = mask_bin:repeatTensor(1, 1, 1, 1)
    else
        local input_float = input:narrow(2, 1, self.real_c)
        local mask_float = input:narrow(2, self.c, 1)
        local mask_bin = binary_mask(mask_float:squeeze(), self.threshold)
        self.ex_mask = mask_bin:repeatTensor(1, self.real_c, 1, 1)
        local input_masked = torch.cmul(input_float, self.ex_mask:cuda())

        self.output = input_masked
    end

    return self.output
end


function BinaryLayer:updateGradInput(input, gradOutput)
    if self.c == 1 then
        self.gradInput = torch.Tensor(self.bz, 1, self.h, self.w):typeAs(input):zero()
    else
        local dummy_mask_grad = torch.Tensor(self.bz, 1, self.h, self.w):typeAs(input):zero()
        local gradData = gradOutput
        gradData:cmul(self.ex_mask:cuda())
        self.gradInput = torch.cat(gradData, dummy_mask_grad, 2)
    end

    return self.gradInput
end

function BinaryLayer:clearState()
    self.output = self.output.new()
    self.gradInput = self.gradInput.new()
end

