.PHONY: check fetch-model fmt vet lint test zizmor

check: fmt vet lint test zizmor

fetch-model:
	go run ./internal/cmd/fetchmodel

fmt:
	@out=$$(gofmt -l .); if [ -n "$$out" ]; then echo "gofmt needed:"; echo "$$out"; exit 1; fi

vet:
	go vet ./...

lint:
	go tool -modfile=tools/go.mod staticcheck ./...

test:
	go test -race ./...

zizmor:
	zizmor .
