################################################################################
#
# Build script for project
#
################################################################################

SRCDIRS := textons combine stencilMatrixMultiply localcues noReorthog sPb intervening gPb convert nonmax damascene postprocess

.PHONY: subdirs $(SRCDIRS)
subdirs: $(SRCDIRS)
$(SRCDIRS):
	$(MAKE) -C $@

convert: stencilMatrixMultiply
combine: stencilMatrixMultiply
textons: convert
noReorthog: stencilMatrixMultiply
sPb: noReorthog
intervening: convert
gPb: convert
nonmax: convert
damascene: textons combine stencilMatrixMultiply localcues noReorthog sPb intervening gPb convert nonmax postprocess


.PHONY: clean
clean:
	rm -rf bin/
	rm -rf obj/

################################################################################
# Rules and targets

# include ../common.mk
