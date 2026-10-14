"""Native Flower App entry points for the maintained FiBA core."""
import shlex
from pathlib import Path
from flwr.client import ClientApp
from flwr.server import ServerApp
from .native_transport import PartitionClient
from .main import parser, run


def options(context):
    return shlex.split(str(context.run_config["options"]))


def client_fn(context):
    argv = options(context)
    cid = int(context.node_config["partition-id"])
    args = parser().parse_args(argv)
    # Resolve before the factory enters the run-owned work directory.
    for name in ('output', 'data_root', 'encoder_checkpoint', 'lira_generator'):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    return PartitionClient(cid, lambda: run(args, client_id=cid), args.seed,
                           args.num_clients, work=args.output.resolve() / "work", framework="torch")

client_app = ClientApp(client_fn=client_fn)
server_app = ServerApp()


@server_app.main()
def server_main(driver, context):
    run(parser().parse_args(options(context)), driver=driver)
